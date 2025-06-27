import os
import glob
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

# =====================
# Regression Head
# =====================
class RegressionHead(nn.Module):
    def __init__(self, hidden_size=768):
        self.hidden_size = hidden_size
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)  # Predict scalar per token

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch, seq, hidden)
        out = self.linear(hidden_states).squeeze(-1)  # (batch, seq)
        
        # mask out padding tokens
        if attention_mask is not None:
            out = out * attention_mask

        # sigmoid to get importance (0-1)
        out = torch.sigmoid(out)

        return out

    def load_weights(self, path):
        linear_weights = torch.load(os.path.join(path, "linear.pt"))
        self.linear.weight.data = linear_weights['weight'].to(self.linear.weight.device)
        self.linear.bias.data = linear_weights['bias'].to(self.linear.bias.device)
        return self

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.linear.state_dict(), os.path.join(path, "linear.pt"))

class ImportanceModel(nn.Module):
    def __init__(self, base_model, regression_head):
        super().__init__()
        self.base_model = base_model
        self.regression_head = regression_head

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return self.regression_head(hidden_states, attention_mask)

    @classmethod
    def from_pretrained(cls, path):
        base_model = AutoModelForCausalLM.from_pretrained(os.path.join(path, "base_model"))
        regression_head = RegressionHead()
        regression_head.load_weights(os.path.join(path, "regression_head"))
        return cls(base_model, regression_head)

# =====================
# Dataset
# =====================
class ImportanceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = self.tokenizer(row['completion_stub'], truncation=True, max_length=self.max_length, return_tensors='pt')

        # For each row, we need: reward_A, reward_B, reward_group_mean, reward_group_var
        # These are scalars, but we want to predict f_theta(t) for each token
        # We'll return the tokens and the rewards for loss computation
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'reward_A': torch.tensor(row['reward_A'], dtype=torch.float32),
            'reward_B': torch.tensor(row['reward_B'], dtype=torch.float32),
            'reward_group_mean': torch.tensor(row['reward_group_mean'], dtype=torch.float32),
            'reward_group_var': torch.tensor(row['reward_group_var'], dtype=torch.float32),
            'split_token_idx': int(row['split_token_idx']),

            'completion_stub': row['completion_stub'],
        }

def collate_fn(batch):
    '''
    Collate function for the dataset.
    Args:
        batch: list of dicts, each containing the keys 'input_ids', 'attention_mask', 'reward_A', 'reward_B', 'reward_group_mean', 'reward_group_var', 'split_token_idx'
    Returns:
        dict with keys 'input_ids', 'attention_mask', 'reward_A', 'reward_B', 'reward_group_mean', 'reward_group_var', 'split_token_idx'
    '''
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # Ensure input_ids are long/int dtype for embedding layer
    input_ids = input_ids.long()
    attention_mask = attention_mask.long()
    
    reward_A = torch.stack([item['reward_A'] for item in batch])
    reward_B = torch.stack([item['reward_B'] for item in batch])
    reward_group_mean = torch.stack([item['reward_group_mean'] for item in batch])
    reward_group_var = torch.stack([item['reward_group_var'] for item in batch])
    split_token_idx = torch.tensor([item['split_token_idx'] for item in batch], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'reward_A': reward_A,
        'reward_B': reward_B,
        'reward_group_mean': reward_group_mean,
        'reward_group_var': reward_group_var,
        'split_token_idx': split_token_idx,
        'completion_stub': [item['completion_stub'] for item in batch],
    }

# =====================
# Loss Function (Likelihood)
# =====================
def nll_loss_pair(R1, R2, m_q, v_q, f_theta):
    '''
    Compute the negative log-likelihood of (R1, R2) under a bivariate normal distribution with mean m_q and covariance v_q * [[1, f_theta], [f_theta, 1]]

    Args:
        R1, R2: (batch,) (rewards)
        m_q: (batch,) (mean of the bivariate normal distribution)
        v_q: (batch,) (variance of the bivariate normal distribution)
        f_theta: (batch,) (importance weights)

    Returns:
        loss: (batch,) (negative log-likelihood)
    '''
    eps = 1e-6
    f_theta = torch.clamp(f_theta, -1 + eps, 1 - eps)  # for stability

    cov = v_q[:, None, None] * torch.stack([
        torch.stack([torch.ones_like(f_theta), f_theta], dim=-1),
        torch.stack([f_theta, torch.ones_like(f_theta)], dim=-1)
    ], dim=-2)  # (batch, 2, 2)

    cov_inv = torch.linalg.inv(cov + eps * torch.eye(2, device=cov.device))  # (batch, 2, 2)
    logdet = torch.logdet(cov + eps * torch.eye(2, device=cov.device))  # (batch,)

    # Compute log-likelihood of (R1, R2) under N([m_q, m_q], v_q * [[1, f_theta], [f_theta, 1]])
    diff = torch.stack([R1 - m_q, R2 - m_q], dim=-1).unsqueeze(-1)  # (batch, 2, 1)
    
    # (batch, 1, 2) @ (batch, 2, 2) @ (batch, 2, 1) = (batch, 1, 1)
    quad = torch.matmul(diff.transpose(-2, -1), torch.matmul(cov_inv, diff)).squeeze(-1).squeeze(-1)  # (batch,)
    
    log_prob = -0.5 * (2 * np.log(2 * np.pi) + logdet + quad) # (batch,)

    return -log_prob

# =====================
# Main Training Script
# =====================
def main():
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(start_time))
    print(f"Start time: {start_time_str}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--finetune_type', type=str, choices=['full', 'lora'], default='full')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--group_size', type=int, default=16)
    parser.add_argument('--rollout_model_name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--rollout_task_name', type=str, default='gsm8k')
    parser.add_argument('--output_dir', type=str, default='importance_networks')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fine_progress_bar', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()


    finetune_type_str = args.finetune_type if args.finetune_type == 'full' else f"lora_{args.lora_rank}"

    # ========== Data Loading ==========
    rollout_model_short = args.rollout_model_name.split("/")[-1]

    parquet_dir = os.path.join("rollout_dump", args.rollout_task_name, rollout_model_short)
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, f"rollouts_{args.group_size}_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir} for group_size {args.group_size}.")
    
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    # Only keep required columns
    df = df[["completion_stub", "reward_A", "reward_B", "reward_group_mean", "reward_group_var", "split_token_idx"]]


    # Get metadata
    with open(os.path.join(parquet_dir, f"rollouts_{args.group_size}_metadata.json"), "r") as f:
        metadata = json.load(f)

    # ========== Tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ========== Dataset ==========
    dataset = ImportanceDataset(df, tokenizer, max_length=metadata['max_tokens'])
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    
    # ========== Model ==========
    config = AutoConfig.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config)
    hidden_size = config.hidden_size
    regression_head = RegressionHead(hidden_size)

    if args.finetune_type == 'lora':
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        base_model = get_peft_model(base_model, lora_config)

    # Enable gradient checkpointing to save memory
    base_model.gradient_checkpointing_enable()

    model = ImportanceModel(base_model, regression_head).to('cuda')

    # ========== Optimizer ==========
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # ========== wandb ==========
    run_name = f"base_{args.base_model.split('/')[-1]}_ft_{finetune_type_str}_roll_{rollout_model_short}_task_{args.rollout_task_name}_G{args.group_size}_lr{args.lr}"
    if not args.disable_wandb:
        wandb.init(entity="sam-bowyer-bristol", project="importance_networks", name=run_name)

    # ========== Training Loop ==========
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", miniters=0 if args.fine_progress_bar else len(train_loader)//10):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            
            pred = model(input_ids, attention_mask)  # (batch, seq)

            # For each example, select f_theta at split_token_idx
            try:
                f_theta = torch.stack([pred[i, idx-1] for i, idx in enumerate(batch['split_token_idx'])])
            except IndexError:
                # Can happen if len(completion_stub_tokens) != split_token_idx
                # (WHICH SHOULD NEVER HAPPEN (!?))
                # breakpoint()
                continue
            
            loss = nll_loss_pair(batch['reward_A'].to('cuda'), batch['reward_B'].to('cuda'), batch['reward_group_mean'].to('cuda'), batch['reward_group_var'].to('cuda'), f_theta).mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * input_ids.size(0)
            
            if not args.disable_wandb:
                wandb.log({'train/loss': loss.item()})

        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", miniters=0 if args.fine_progress_bar else len(val_loader)//10):
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')

                outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                pred = model.regression_head(hidden_states, attention_mask)
                
                try:
                    f_theta = torch.stack([pred[i, idx-1] for i, idx in enumerate(batch['split_token_idx'])])
                except IndexError:
                    # Can happen if len(completion_stub_tokens) != split_token_idx
                    # (WHICH SHOULD NEVER HAPPEN (!?))
                    # breakpoint()
                    continue
                
                loss = nll_loss_pair(batch['reward_A'].to('cuda'), batch['reward_B'].to('cuda'), batch['reward_group_mean'].to('cuda'), batch['reward_group_var'].to('cuda'), f_theta).mean()
                
                val_loss += loss.item() * input_ids.size(0)
        
        val_loss /= len(val_loader.dataset)

        if not args.disable_wandb:
            wandb.log({'val/loss': val_loss})

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # ========== Save ==========
    save_dir = os.path.join(args.output_dir, args.base_model.split('/')[-1], f"rollouts_{rollout_model_short}_{args.rollout_task_name}_{finetune_type_str}_G{args.group_size}_lr{args.lr}")
    os.makedirs(save_dir, exist_ok=True)

    model.base_model.save_pretrained(os.path.join(save_dir, "base_model"))
    model.regression_head.save_pretrained(os.path.join(save_dir, "regression_head"))
    # torch.save(model.state_dict(), os.path.join(save_dir, 'importance_network.pt'))
    
    if not args.disable_wandb:
        wandb.finish()
    print(f"Model saved to {save_dir}")

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_str}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    torch.backends.cuda.preferred_linalg_library('magma')
    main()
