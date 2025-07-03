import argparse
import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_importance_network import ImportanceModel
from transformers import AutoTokenizer
from text_heatmap import render_text_heatmap_matplotlib, render_text_heatmap_terminal

from groundtruth_f_theta import get_groundtruth_f_theta
from vllm import LLM

plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

# # Set font sizes
# plt.rc('font', size=8)          # controls default text sizes
# plt.rc('axes', titlesize=10)    # fontsize of the axes title
# plt.rc('axes', labelsize=9)     # fontsize of the x and y labels
# plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
# plt.rc('legend', fontsize=7)    # legend fontsize
# plt.rc('figure', titlesize=9)   # fontsize of the figure title


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance_network_dir", type=str, default="importance_networks")
    parser.add_argument("--importance_network_base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--importance_network_ft_type", type=str, default="full")
    parser.add_argument("--importance_network_ft_lr", type=float, default=1e-3)
    parser.add_argument("--importance_network_rollout_model_short", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--importance_network_rollout_task_name", type=str, default="gsm8k")
    parser.add_argument("--importance_network_group_size", type=int, default=16)
    parser.add_argument("--rollout_model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rollout_task_name", type=str, default="gsm8k")
    parser.add_argument("--rollout_file_number", type=int, default=None)
    parser.add_argument("--rollout_group_size", type=int, default=16)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--num_pairs", type=int, default=2)
    parser.add_argument("--get_true_f_theta", action="store_true", default=False)
    parser.add_argument("--plot_true_f_theta_comparison", action="store_true", default=False)
    parser.add_argument("--eager", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # ========== Get Model Names ==========
    importance_network_base_model_short = args.importance_network_base_model.split("/")[-1]
    importance_network_rollout_model_short = args.importance_network_rollout_model_short.split("/")[-1]
    rollout_model_short = args.rollout_model_name.split("/")[-1]
    importance_network_dir = os.path.join(args.importance_network_dir, importance_network_base_model_short, f"rollouts_{importance_network_rollout_model_short}_{args.importance_network_rollout_task_name}_{args.importance_network_ft_type}_G{args.importance_network_group_size}_lr{args.importance_network_ft_lr}")

    # ========== Rollout Data ==========
    parquet_dir = os.path.join("rollout_dump", args.rollout_task_name, rollout_model_short)
    if args.rollout_file_number is None:
        pattern = os.path.join(parquet_dir, f"rollouts_{args.rollout_group_size}_*.parquet")
        parquet_files = sorted(glob.glob(pattern))
        if not parquet_files:
            print(f"No matching parquet files found in {parquet_dir} for group_size {args.rollout_group_size}.")
            return
        parquet_file = parquet_files[0]  # Pick the first matching file
    else:
        parquet_file = os.path.join(parquet_dir, f"rollouts_{args.rollout_group_size}_{args.rollout_file_number}.parquet")
        if not os.path.exists(parquet_file):
            print(f"Parquet file {parquet_file} does not exist.")
            return
    
    print(f"Loading parquet file {parquet_file}...\n")
    rollout_data = pd.read_parquet(parquet_file)

    # ========== Importance Network ==========
    print(f"Loading importance network from {importance_network_dir}...\n")
    importance_network_model = ImportanceModel.from_pretrained(importance_network_dir).to('cuda')
    importance_network_model.eval()

    # ========== Get Completion Tokens ==========
    tokenizer = AutoTokenizer.from_pretrained(args.rollout_model_name)

    prompts = []
    completions = []
    completion_stubs = []
    group_reward_variances = []

    # Group by prompt_id and completion_text_{A,B}
    prompt_groups = rollout_data.groupby(["prompt_id"])
    for (prompt_id), group in list(prompt_groups)[:args.num_prompts]:
        # For each group_id (should be the same for all rows in this group)
        for i, row in group.head(args.num_pairs).iterrows():
            prompt = row['prompt_text']
            completion_text_A = row['completion_text_A']
            completion_text_B = row['completion_text_B']

            prompts.extend([prompt]*2)
            completions.extend([completion_text_A, completion_text_B])
            completion_stubs.extend([row['completion_stub']]*2)
            group_reward_variances.extend([row['reward_group_var']]*2)

    completion_tokens = [tokenizer.encode(completion) for completion in completions]
    completion_tokens_str = [[tokenizer.decode(token_id) for token_id in token_ids] for token_ids in completion_tokens]

    # pad all completion tokens to the same length
    max_length = max(len(token_ids) for token_ids in completion_tokens)
    completion_tokens = [token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids)) for token_ids in completion_tokens]
    completion_tokens = torch.tensor(completion_tokens)
    attention_mask = torch.ones_like(completion_tokens)

    # ========== Get Importance Scores ==========
    with torch.no_grad():
        importance_scores = importance_network_model(completion_tokens.to('cuda'), attention_mask.to('cuda')).cpu().numpy()

    # Apply some function to the importance scores
    # max_importance_score = np.max(importance_scores)
    # importance_scores = np.exp(importance_scores - max_importance_score)
    # importance_scores = np.log(importance_scores)


    # ======= Setup for Groundtruth f_theta =======
    if args.get_true_f_theta:
        print(f"Loading VLLM model for groundtruth f_theta computation...\n")
        # Load model with VLLM
        model = LLM(
            model=args.rollout_model_name,
            enable_prefix_caching=True,
            dtype=torch.bfloat16,
            enforce_eager=args.eager,
        )

        # Set up tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.rollout_model_name)
        EOS_TOKEN_ID = tokenizer.eos_token_id
        EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # ========== Print Importance Scores ==========
    print()
    for i in range(len(completion_tokens_str)):
        if i % (args.num_pairs*2) == 0:
            print("#"*100 + f"\nPrompt {i//(args.num_pairs*2)}\n")

        if i % 2 == 0:
            print("-"*100)
            print(f"Completion Stub {(i%(args.num_pairs*2))//(args.num_pairs)}: {completion_stubs[i//2]}\n")

        print(f"Completion {i%(args.num_pairs*2)}\n")

        if args.get_true_f_theta:
            true_f_theta = get_groundtruth_f_theta(
                prompts[i],
                completions[i],
                group_reward_variances[i],
                args.rollout_task_name,
                model,
                tokenizer,
            )
            print(f"TRUE f_theta: {true_f_theta[:len(completion_tokens_str[i])].numpy()}\n")

            if args.plot_true_f_theta_comparison:
                plt.figure(figsize=(len(completion_tokens_str[i])/4, 5))
                plt.plot(true_f_theta[:len(completion_tokens_str[i])].numpy(), label="True f_theta")
                plt.plot(importance_scores[i][:len(completion_tokens_str[i])], label="Importance Network f_theta")
                plt.legend()

                # breakpoint()

                # put tokens on the x-axis
                plt.xticks(
                    range(len(completion_tokens_str[i])),
                    [t.replace("$", "\$") for t in completion_tokens_str[i]],
                    rotation=45
                )

                plt.tight_layout()

                # plt.show()
                plt.savefig(f"plots/true_f_theta_comparison_{i}.png")
                plt.close()
        
        print(f"f_theta: {importance_scores[i][:len(completion_tokens_str[i])]}\n")

        # render_text_heatmap_matplotlib(completion_tokens_str[i], importance_scores[i][:len(completion_tokens_str[i])], f"plots/importance_scores_{i}.pdf")
        actual_score = importance_scores[i][:len(completion_tokens_str[i])]
        actual_score = [actual_score[i+1] - actual_score[i] for i in range(len(actual_score)-1)]
        
        render_text_heatmap_terminal(completion_tokens_str[i], actual_score, rescale_value=False, num_pad_end=1)

        print()




if __name__ == "__main__":
    main()
