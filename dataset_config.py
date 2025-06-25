import os
from typing import Any, Dict, List
from transformers import AutoTokenizer

# To use a new dataset, you can add its configuration to the DATASET_CONFIGS dictionary.
# Each configuration should specify:
# - 'hf_path': The path to the dataset on Hugging Face Hub.
# - 'split': The dataset split to use (e.g., 'train').
# - 'short_name': A short name for the dataset to be used in run names.
# - 'preprocess_fnc': A function that takes a dataset example and a tokenizer,
#   and returns a dictionary with 'input_ids' and other relevant fields.
#
# The `preprocess_fnc` is crucial as different datasets have different structures.
# For example, the 'countdown' dataset requires constructing a prompt from 'numbers' and 'target' fields,
# while 'gsm8k' involves formatting a 'question' field.
# You can define a new preprocessing function for each new dataset you add.

def preprocess_countdown(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Preprocess a single example from the Countdown-Tasks-3to4 dataset.
    
    Args:
        example (Dict[str, Any]): A single example from the dataset.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        Dict[str, Any]: A dictionary with 'prompt' and 'input_ids' keys.
    """

    SYSTEM_MESSAGE = (
        "You are a helpful assistant. You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    PROMPT_TEMPLATE = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final equation and answer in "
        "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
    )
    
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(
        prefix, tokenize=True, continue_final_message=True
    )
    prompt = tokenizer.decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {"prompt": prompt, "input_ids": input_ids}


def preprocess_gsm8k(example : Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Preprocess a single example from the gsm8k dataset.
    
    Args:
        example (Dict[str, Any]): A single example from the dataset.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        Dict[str, Any]: A dictionary with 'prompt' and 'input_ids' keys.
    """

    SYSTEM_MESSAGE = (
        "You are a helpful assistant. You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    PROMPT_TEMPLATE = (
        "Question: {question}\n\n"
        "Think step-by-step inside <think>...</think> tags. "
        "Then, give your final numerical answer inside <answer>...</answer> tags."
    )
    
    user_content = PROMPT_TEMPLATE.format(question=example["question"])
    
    chat = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, continue_final_message=True
    )
    
    prompt = tokenizer.decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {"prompt": prompt, "input_ids": input_ids}


DATASET_CONFIGS = {
    "countdown": {
        "hf_path": "Jiayi-Pan/Countdown-Tasks-3to4",
        "config_name": "default",
        "split": "train",
        "short_name": "cd",
        "preprocess_fnc": preprocess_countdown,
        "num_rows": 10_000, # use first 10k rows
    },
    "gsm8k": {
        "hf_path": "gsm8k",
        "config_name": "main",
        "split": "train",
        "short_name": "gsm",
        "preprocess_fnc": preprocess_gsm8k,
        "num_rows": None, # use all rows
    },
}

def get_dataset_config(name: str):
    """Retrieve the configuration for a given dataset by its short name."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name] 