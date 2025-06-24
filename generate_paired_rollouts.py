import os
from pathlib import Path

import argparse
import time 
import gc

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import torch
import numpy as np
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import wandb
WANDB_ENTITY = "sam-bowyer-bristol"
WANDB_PROJECT = "paired_rollouts"

from rewards import compute_reward, TASK2PARTIAL_REWARD_NAMES
from dataset_config import get_dataset_config


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--group_size", type=int,  default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="rollout_dump")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    MODEL_NAME_SHORT = args.model_name.split("/")[-1]


    # Load model with VLLM
    model = LLM(
        model=args.model_name,
        enable_prefix_caching=True,
        dtype=torch.bfloat16,
        enforce_eager=True,
    )

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # Load task -- ONLY TRAINING DATA
    dataset_config = get_dataset_config(args.task_name)
    dataset = load_dataset(dataset_config["hf_path"], split=dataset_config["split"], name=dataset_config["config_name"])
    dataset = dataset.map(dataset_config["preprocess_fnc"], fn_kwargs={"tokenizer": tokenizer})

    # Create output directory
    output_dir = Path(args.output_dir) / args.task_name / MODEL_NAME_SHORT
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a metadata file
    metadata_path = output_dir / f"rollouts_{args.group_size}_metadata.json"
    metadata = {
        "model_name": args.model_name,
        "task_name": args.task_name,
        "group_size": args.group_size,
        "num_epochs": args.num_epochs,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_group_id": -1, # will be incremented by 1 for each new group
    }
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    # Load partial reward names for the task
    partial_reward_names = TASK2PARTIAL_REWARD_NAMES[args.task_name]

    # Set up the parquet file schema
    schema = pa.schema([
        pa.field('prompt_id', pa.int64()),
        pa.field('prompt_text', pa.string()),
        pa.field('group_id', pa.int64()),
        pa.field('completion_id', pa.int64()),
        pa.field('split_token_idx', pa.int64()),
        pa.field('completion_text', pa.string()),
        pa.field('reward', pa.float64()),
        pa.field('reward_group_mean', pa.float64()),
        pa.field('reward_group_var', pa.float64()),
        *[pa.field(f"{reward_name}", pa.float64()) for reward_name in partial_reward_names],
        *[pa.field(f"{reward_name}_group_mean", pa.float64()) for reward_name in partial_reward_names],
        *[pa.field(f"{reward_name}_group_var", pa.float64()) for reward_name in partial_reward_names],
    ])

    # Initialize the parquet file
    output_file = output_dir / f"rollouts_{args.group_size}.parquet"

    # Write the initial (empty) file with the defined schema
    if not os.path.exists(output_file):
        initial_df = pd.DataFrame(columns=schema.names)
        table = pa.Table.from_pandas(initial_df, schema=schema)
        pq.write_table(table, output_file)
        print(f"Initialized empty Parquet file: {output_file}")

    else:
        print(f"Parquet file already exists: {output_file}")

        # Load the existing metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Set current group_id to the max_group_id
    group_id = metadata["max_group_id"]

    # Set up wandb
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{MODEL_NAME_SHORT}_{args.task_name}_G{args.group_size}_E{args.num_epochs}_maxT{args.max_tokens}_temp{args.temperature}",
    )

    wandb_stats = {
        "group_id": [],

        "split_token_idx_group_mean": [],
        "split_token_idx_group_var": [],

        "reward_group_mean": [],
        "reward_group_var": [],

        "completion_length_group_mean": [],
        "completion_length_group_var": [],

        "paired_rollout_time": [],
    }

    # Create a new table to store the rollouts
    new_rollouts = {k: [] for k in schema.names}

    # Now we're ready to generate the rollouts
    for epoch in range(args.num_epochs):
        print(f"Generating rollouts for epoch {epoch+1} of {args.num_epochs}")
        
        # Process the dataset
        for i, row in enumerate(dataset):
            group_start_time = time.time()

            # Get the prompt text
            prompt_text = row["prompt"]
            prompt_id = i

            # Increment the group_id
            group_id += 1

            # Generate the even-numbered rollouts
            even_rollouts = model.generate(
                prompts=[prompt_text],
                sampling_params=SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=args.group_size//2,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
            )

            # Truncate the even-numbered rollouts to create pairing
            # First, sample split_idx for each even-numbered rollout uniformly from [0, len(even_rollout_i))
            split_idxs = np.random.randint(low=0, high=[len(r.text) for r in even_rollouts[0].outputs])

            # Then, truncate the even-numbered rollouts at the corresponding split_idx
            truncated_even_rollouts_text = [r.text[:split_idxs[i]] for i, r in enumerate(even_rollouts[0].outputs)]

            assert len(even_rollouts[0].outputs) == args.group_size//2, "Number of even-numbered rollouts is not equal to group_size/2"

            # Now, generate the odd-numbered rollouts
            odd_rollouts = model.generate(
                prompts=truncated_even_rollouts_text, # len(truncated_even_rollouts_text) == args.group_size//2 ?
                sampling_params=SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=1,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
            )

            # TODO: check finish reasons for rollouts
            
            # Get the completion text -- TODO: check if this is correct
            even_rollouts_text = [r.text for r in even_rollouts[0].outputs]
            odd_rollouts_text = [r.outputs[0].text for r in odd_rollouts]

            # Finally, interleave the even and odd-numbered rollouts
            paired_rollouts = [r for pair in zip(even_rollouts_text, odd_rollouts_text) for r in pair]

            # Now, compute the rewards
            rewards_with_partials = [compute_reward(r, row, args.task_name, EOS_TOKEN) for r in paired_rollouts]

            # Get the rewards
            rewards = [r[0] for r in rewards_with_partials]
            partial_rewards = [r[1] for r in rewards_with_partials]

            # And mean/var of the rewards
            mean_reward = np.mean(rewards)
            var_reward = np.var(rewards)

            partial_rewards_mean = {reward_name: np.mean([r[reward_name] for r in partial_rewards]) for reward_name in partial_reward_names}
            partial_rewards_var = {reward_name: np.var([r[reward_name] for r in partial_rewards]) for reward_name in partial_reward_names}

            # Update the new_rollouts
            new_rollouts["prompt_id"].extend([prompt_id] * args.group_size)
            new_rollouts["prompt_text"].extend([prompt_text] * args.group_size)
            new_rollouts["group_id"].extend([group_id] * args.group_size)
            new_rollouts["completion_id"].extend([i] * args.group_size)
            new_rollouts["split_token_idx"].extend([idx_copy for idx in split_idxs for idx_copy in [idx, idx]])
            new_rollouts["completion_text"].extend(paired_rollouts)
            new_rollouts["reward"].extend(rewards)
            new_rollouts["reward_group_mean"].extend([mean_reward] * args.group_size)
            new_rollouts["reward_group_var"].extend([var_reward] * args.group_size)
            for reward_name in partial_reward_names:
                new_rollouts[reward_name].extend([r[reward_name] for r in partial_rewards])
                new_rollouts[f"{reward_name}_group_mean"].extend([partial_rewards_mean[reward_name]] * args.group_size)
                new_rollouts[f"{reward_name}_group_var"].extend([partial_rewards_var[reward_name]] * args.group_size)

            # Update the wandb stats
            wandb_stats["group_id"].append(group_id)
            wandb_stats["split_token_idx_group_mean"].append(np.mean(split_idxs))
            wandb_stats["split_token_idx_group_var"].append(np.var(split_idxs))
            wandb_stats["reward_group_mean"].append(mean_reward)
            wandb_stats["reward_group_var"].append(var_reward)
            wandb_stats["completion_length_group_mean"].append(np.mean([len(r) for r in paired_rollouts]))
            wandb_stats["completion_length_group_var"].append(np.var([len(r) for r in paired_rollouts]))
            wandb_stats["paired_rollout_time"].append(time.time() - group_start_time)


            # Log the wandb stats
            wandb.log({k: v[-1] for k, v in wandb_stats.items()})

            # Append the new rollouts to the parquet file
            if len(new_rollouts["prompt_id"]) > 50_000:
                dump_rollouts_to_parquet(pd.DataFrame(new_rollouts), output_file, schema)
                new_rollouts = {k: [] for k in schema.names}

                metadata["max_group_id"] = group_id
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

            # if i > 0:
            #     break

    # Dump the remaining rollouts
    dump_rollouts_to_parquet(pd.DataFrame(new_rollouts), output_file, schema)

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Close wandb
    wandb.finish()

    # clean up
    # del rollouts_df
    # del new_rollouts_table
    # gc.collect()


def dump_rollouts_to_parquet(rollouts_df, output_file, schema):
    if len(rollouts_df) == 0:
        return
    
    new_rollouts_table = pa.Table.from_pandas(rollouts_df, schema=schema)

    # add max_group_id to the filename
    output_file = output_file.with_name(f"{output_file.stem}_{rollouts_df['group_id'].max()}.parquet")

    with pq.ParquetWriter(output_file, schema=schema, compression='snappy') as writer:
        writer.write_table(new_rollouts_table)

    print(f"Appended {len(rollouts_df)} records to {output_file}")

    # free up memory
    # del rollouts_df
    # del new_rollouts_table
    # gc.collect()
    


if __name__ == "__main__":
    main()