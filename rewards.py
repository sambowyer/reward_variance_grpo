import re
from typing import Any, Dict, List, Tuple


def format_reward_func(completion: str, EOS_TOKEN: str, **kwargs) -> float:
    """
    This function is used to reward the model for following the format of the prompt.
    It checks that the model has included a <think> tag and a <answer> tag.
    It also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, prompt: Dict[str, Any], **kwargs) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """

    # Extract the numbers and target from the prompt
    nums = prompt["nums"]
    target = prompt["target"]

    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0

def gsm8k_correctness_reward_func(completion: str, prompt: Dict[str, Any], **kwargs) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer
    """
    try:
        correct_answer = re.search(r"####\s*(-?\d+(\.\d+)?)", prompt["answer"])
        if correct_answer:
            correct_answer = float(correct_answer.group(1))
        
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match:
            completion_answer = float(match.group(1))
        else:
            return 0.0

        # Check if the completion is correct
        if abs(completion_answer - correct_answer) < 1e-5:
            return 1.0
        return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(
    completion: str,
    prompt: Dict[str, Any],
    task_name: str,
    EOS_TOKEN: str
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the reward for a given completion.

    Args:
        completion (str): The completion to evaluate
        prompt (Dict[str, Any]): The prompt being evaluated
        task_name (str): The name of the task
        EOS_TOKEN (str): The end of sequence token

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing the reward (float) and metrics (dict of partial-rewards)
    """
    # Compute the partial rewards
    partial_reward_names = TASK2PARTIAL_REWARD_NAMES[task_name]
    partial_rewards = {}

    for partial_reward_name in partial_reward_names:
        partial_rewards[partial_reward_name] = PARTIAL_REWARD_NAME2FUNC[partial_reward_name](completion=completion, prompt=prompt, EOS_TOKEN=EOS_TOKEN)

    # Compute the overall reward
    reward = sum(partial_rewards.values())

    return reward, partial_rewards






PARTIAL_REWARD_NAME2FUNC = {
    "format_reward": format_reward_func,
    "correctness_reward": equation_reward_func,
    "gsm8k_correctness_reward": gsm8k_correctness_reward_func,
}

TASK2PARTIAL_REWARD_NAMES = {
    "gsm8k": ["format_reward", "gsm8k_correctness_reward"],
    "math": ["format_reward", "correctness_reward"],
    "countdown": ["format_reward", "correctness_reward"],
}