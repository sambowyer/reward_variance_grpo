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
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(start_time))
    print(f"Start time: {start_time_str}")

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--group_size", type=int,  default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="rollout_dump")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eager", action="store_true", default=False)
    args = parser.parse_args()

    print(f"Args: {args}\n")

    assert args.group_size % 2 == 0, "Group size must be even"
    
    # Set up some constants
    MODEL_NAME_SHORT = args.model_name.split("/")[-1]
    NUM_PAIRS = args.group_size // 2

    # Create output directory (if it doesn't exist already)
    output_dir = Path(args.output_dir) / args.task_name / MODEL_NAME_SHORT
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a metadata file (if it doesn't exist already)
    metadata_path = output_dir / f"rollouts_{args.group_size}_metadata.json"
    base_metadata = {
        "model_name": args.model_name,
        "task_name": args.task_name,
        "group_size": args.group_size,
        "num_epochs": 0, # will be incremented by 1 for each completed epoch
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_group_id": -1, # will be incremented by 1 for each new group
        "file_count": 0, # will be incremented by 1 for each new file saved
    }

    # Initialize the metadata file
    if not os.path.exists(metadata_path) or args.overwrite:
        with open(metadata_path, "w") as f:
            json.dump(base_metadata, f, indent=4)
        
        metadata = {**base_metadata}
    else:
        # If the metadata file already exists, load it and check our args against it
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        for k, v in base_metadata.items():
            if k in ("max_group_id", "file_count", "num_epochs"):
                continue # these are not experiment-specific metadata

            # If the metadata file already exists and the args are different, raise an error
            if k not in metadata or metadata[k] != v:
                raise ValueError(f"Metadata file {metadata_path} already exists and has different values for {k} ({metadata[k]}) than the current args ({v}). To overwrite, use --overwrite.")


    # Set current group_id to the max_group_id (it will be incremented by 1 for each new group)
    group_id = metadata["max_group_id"]

    # Set file_count to the file_count in the metadata
    file_count = metadata["file_count"]

    # Load partial reward names for the task
    partial_reward_names = TASK2PARTIAL_REWARD_NAMES[args.task_name]

    # Set up the parquet file schema
    schema = pa.schema([
        pa.field('prompt_id', pa.int64()),
        pa.field('prompt_text', pa.string()),
        pa.field('group_id', pa.int64()),
        pa.field('completion_pair_id', pa.int64()),
        pa.field('split_token_idx', pa.int64()),
        pa.field('completion_stub', pa.string()),
        pa.field('completion_text_A', pa.string()),
        pa.field('completion_text_B', pa.string()),
        pa.field('reward_A', pa.float64()),
        pa.field('reward_B', pa.float64()),
        pa.field('reward_group_mean', pa.float64()),
        pa.field('reward_group_var', pa.float64()),
        *[pa.field(f"{reward_name}_A", pa.float64()) for reward_name in partial_reward_names],
        *[pa.field(f"{reward_name}_B", pa.float64()) for reward_name in partial_reward_names],
        *[pa.field(f"{reward_name}_group_mean", pa.float64()) for reward_name in partial_reward_names],
        *[pa.field(f"{reward_name}_group_var", pa.float64()) for reward_name in partial_reward_names],
    ])

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

    # Load model with VLLM
    model = LLM(
        model=args.model_name,
        enable_prefix_caching=True,
        dtype=torch.bfloat16,
        enforce_eager=args.eager,
    )

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # Load task -- ONLY TRAINING DATA
    dataset_config = get_dataset_config(args.task_name)
    dataset = load_dataset(dataset_config["hf_path"], split=dataset_config["split"], name=dataset_config["config_name"])
    if dataset_config["num_rows"] is not None:
        dataset = dataset.select(range(dataset_config["num_rows"]))
    dataset = dataset.map(dataset_config["preprocess_fnc"], fn_kwargs={"tokenizer": tokenizer})

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

            # Generate the even-numbered rollouts (A)
            even_rollouts = model.generate(
                prompts=[prompt_text],
                sampling_params=SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=NUM_PAIRS,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
            )

            assert len(even_rollouts[0].outputs) == NUM_PAIRS, "Number of even-numbered rollouts is not equal to group_size/2"

            # Truncate the even-numbered rollouts to create pairing
            # First, sample split_idx for each even-numbered rollout uniformly from [0, len(even_rollout_tokens_i)-1)
            split_idxs = np.random.randint(low=0, high=[max(len(r.token_ids)-1, 1) for r in even_rollouts[0].outputs])

            # Then, truncate the even-numbered rollouts at the corresponding split_idx (IN TOKEN-SPACE)
            truncated_even_rollouts_token_ids = [r.token_ids[:split_idxs[i]] for i, r in enumerate(even_rollouts[0].outputs)]

            # Convert the token_ids back to text for vllm
            truncated_even_rollouts_text = [tokenizer.decode(r) for r in truncated_even_rollouts_token_ids]

            # Now, generate the odd-numbered rollouts (B)
            odd_rollouts = model.generate(
                prompts=[prompt_text + r for r in truncated_even_rollouts_text],
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
            odd_rollouts_text = [truncated_even_rollouts_text[i] + r.outputs[0].text for i, r in enumerate(odd_rollouts)]

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
            new_rollouts["prompt_id"].extend([prompt_id] * NUM_PAIRS)
            new_rollouts["prompt_text"].extend([prompt_text] * NUM_PAIRS)
            new_rollouts["group_id"].extend([group_id] * NUM_PAIRS)
            new_rollouts["completion_pair_id"].extend(list(range(NUM_PAIRS)))
            new_rollouts["split_token_idx"].extend(split_idxs.tolist())

            new_rollouts["completion_stub"].extend(truncated_even_rollouts_text)
            new_rollouts["completion_text_A"].extend(even_rollouts_text)
            new_rollouts["completion_text_B"].extend(odd_rollouts_text)

            new_rollouts["reward_A"].extend(rewards[::2])
            new_rollouts["reward_B"].extend(rewards[1::2])
            new_rollouts["reward_group_mean"].extend([mean_reward] * NUM_PAIRS)
            new_rollouts["reward_group_var"].extend([var_reward] * NUM_PAIRS)
            for reward_name in partial_reward_names:
                new_rollouts[f"{reward_name}_A"].extend([r[reward_name] for r in partial_rewards[::2]])
                new_rollouts[f"{reward_name}_B"].extend([r[reward_name] for r in partial_rewards[1::2]])
                new_rollouts[f"{reward_name}_group_mean"].extend([partial_rewards_mean[reward_name]] * NUM_PAIRS)
                new_rollouts[f"{reward_name}_group_var"].extend([partial_rewards_var[reward_name]] * NUM_PAIRS)


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

            # Append the new rollouts to the parquet file if it's getting big
            if len(new_rollouts["prompt_id"]) > 50_000:
                dump_rollouts_to_parquet(pd.DataFrame(new_rollouts), output_dir / f"rollouts_{args.group_size}_{file_count}.parquet", schema)
                new_rollouts = {k: [] for k in schema.names}
                file_count += 1

                # Update the metadata
                metadata["max_group_id"] = group_id
                metadata["file_count"] = file_count
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)

            # if i > 3:
            #     break

        # Increment the number of epochs completed
        metadata["num_epochs"] += 1

    # Dump the remaining rollouts
    dump_rollouts_to_parquet(pd.DataFrame(new_rollouts), output_dir / f"rollouts_{args.group_size}_{file_count}.parquet", schema)
    file_count += 1

    # Save metadata
    metadata["max_group_id"] = group_id
    metadata["file_count"] = file_count
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Close wandb
    wandb.finish()

    # clean up
    # del rollouts_df
    # del new_rollouts_table
    # gc.collect()

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_str}")
    print(f"Total time: {end_time - start_time} seconds")


def dump_rollouts_to_parquet(rollouts_df, output_file, schema):
    if len(rollouts_df) == 0:
        return
    
    new_rollouts_table = pa.Table.from_pandas(rollouts_df, schema=schema)

    with pq.ParquetWriter(output_file, schema=schema, compression='snappy') as writer:
        writer.write_table(new_rollouts_table)

    print(f"Appended {len(rollouts_df)} records to {output_file}")

    # free up memory
    # del rollouts_df
    # del new_rollouts_table
    # gc.collect()
    


if __name__ == "__main__":
    main()