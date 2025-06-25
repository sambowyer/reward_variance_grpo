import argparse
import os
import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-0.6B")
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--group_size", type=int, default=10)
    parser.add_argument("--num_prompts", type=int, default=2)
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Find the matching parquet file
    model_short = args.model_name.split("/")[-1]
    parquet_dir = os.path.join("rollout_dump", args.task_name, model_short)
    pattern = os.path.join(parquet_dir, f"rollouts_{args.group_size}_*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        print(f"No matching parquet files found in {parquet_dir} for group_size {args.group_size}.")
        return
    parquet_file = parquet_files[0]  # Pick the first matching file

    data = pd.read_parquet(parquet_file)

    # Group by prompt_id and prompt_text
    prompt_groups = data.groupby(["prompt_id", "prompt_text"])
    for (prompt_id, prompt_text), group in list(prompt_groups)[:args.num_prompts]:
        print("\n" + "#"*100)
        print(f"\nPrompt {prompt_id}: {prompt_text}")
        # For each group_id (should be the same for all rows in this group)
        for i, row in group.head(args.num_pairs).iterrows():
            print("-"*100)
            print(f"> Pair {row['completion_pair_id']}:")
            print(f">>> completion_stub: {row['completion_stub']}\n")
            print(f">>> completion_text_A: {row['completion_text_A']}\n")
            print(f">>> completion_text_B: {row['completion_text_B']}\n")
            print(f">>> reward_A: {row['reward_A']}\n")
            print(f">>> reward_B: {row['reward_B']}\n")


    if args.debug:
        print("Parquet loaded as variable 'data'.")
        breakpoint()


if __name__ == "__main__":
    main()
