from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import numpy as np
import tqdm
from rewards import compute_reward
from dataset_config import get_dataset_config
from datasets import load_dataset

def get_groundtruth_f_theta(
    prompt: str,
    completion_str: str,
    group_reward_variance: float,
    task_name: str,
    inference_model : LLM,
    tokenizer : AutoTokenizer,
    num_rollouts_per_token: int = 32,
    max_tokens: int = 1024,
    temperature: float = 1.0,
):
    '''
    prompt: str
        The prompt string to get the groundtruth f_theta for.
    completion_str: str
        The completion string to get the groundtruth f_theta for.
    group_reward_variance: float
        The variance of the group reward (v_q) for the specific question.
    task_name: str
        The name of the task.
    inference_model : LLM
        The inference model to get rollouts from.
    tokenizer : AutoTokenizer
        The tokenizer to use to tokenize the completion string.
    num_rollouts_per_token: int = 32
        The number of rollouts to get for each token.
    max_tokens: int = 1024
        The maximum number of tokens to use for the completion string.
    '''
    # Get dataset config
    dataset_config = get_dataset_config(task_name)
    dataset = load_dataset(dataset_config["hf_path"], split=dataset_config["split"], name=dataset_config["config_name"])
    if dataset_config["num_rows"] is not None:
        dataset = dataset.select(range(dataset_config["num_rows"]))
    dataset = dataset.map(dataset_config["preprocess_fnc"], fn_kwargs={"tokenizer": tokenizer})

    # Get answers 
    # i.e. find row in dataset that has the same prompt as the thing we're looking at 
    # (prompt (with chat template) + partial_completion_str)
    dataset_row = next(row for row in dataset if prompt in row['prompt'])


    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # Get tokens
    tokenized_completion = tokenizer(completion_str, return_tensors="pt", padding=False, truncation=True, max_length=max_tokens)
    input_ids = tokenized_completion["input_ids"]
    attention_mask = tokenized_completion["attention_mask"]

    # Create list of prompts with partial completions
    prompts = []
    for t in range(len(input_ids[0])):
        partial_completion_str = tokenizer.decode(input_ids[0, :t+1])
        prompts.append(prompt + partial_completion_str)

    # Get rollouts in batches of 8 prompts at a time
    all_rollouts = []
    for i in tqdm.trange(0, len(prompts), 4, desc="Getting rollouts for groundtruth f_theta"):
        batch_prompts = prompts[i:i+4]
        batch_rollouts = inference_model.generate(
            prompts=batch_prompts,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature, 
                n=num_rollouts_per_token,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
            use_tqdm=False,
        )
        all_rollouts.extend(batch_rollouts)

    f_theta = torch.zeros(len(input_ids[0]), dtype=torch.float32)
    rewards_per_token = torch.zeros((len(input_ids[0]), num_rollouts_per_token), dtype=torch.float32)

    for t in range(len(input_ids[0])):
        # Get the completion text for this position
        rollouts_text = [r.text for r in all_rollouts[t].outputs]

        # Get rewards
        rewards = [compute_reward(r, dataset_row, task_name, EOS_TOKEN) for r in rollouts_text]
        rewards = [r[0] for r in rewards] # ignore partial rewards

        # Get variance of the rewards
        rewards_per_token[t,:] = torch.tensor(rewards)

    per_token_reward_variance = torch.var(rewards_per_token, dim=1)
    overall_reward_variance = torch.var(rewards_per_token.flatten())

    f_theta = 1 - (per_token_reward_variance / overall_reward_variance)

    return f_theta
