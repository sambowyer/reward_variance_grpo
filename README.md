# Reward Variance GRPO

Using per-token expected reward-variance to modify the per-token GRPO loss.

## Conda environment with environment.yml

```bash
conda env create -f environment.yml
```

#### Bluepebble

N.B.: To get it working on bluepebble, I had to run

```
conda install -c nvidia cuda-compiler
```

after the regular env setup.


# Outline

1. Generate just a ton of (paired) rollouts using model X on task Y, save the completions and rewards.
    - 'Paired' because WITHIN a group of $G$ rollouts on a single prompt, we generate $G/2$ rollouts up to some random $t$ th token, then carry on two separate rollouts from this point.
    - Only need to save the completions and rewards, no gradients.
2. Train an 'importance network' $f_\theta$ to predict the expected reward-variance of the rollouts for a given token.
    - This can be done by just putting a head on an (autoregressive) LLM.
    - Train $f_\theta$ by maximum likelihood on the paired rollout rewards:
    $$
    \begin{align}
    P{\begin{pmatrix} R_1 \\ R_2 \end{pmatrix}} &= 
    \mathcal{N}{\begin{pmatrix} m_q \\ m_q \end{pmatrix}, v_q \begin{pmatrix} 1 & f_\theta(t) \\ f_\theta(t) & 1\end{pmatrix}}
    \end{align}
    $$
    - $m_q$ is the mean reward of the rollouts in the group (of size $G$) for the prompt.
    - $v_q$ is the variance of the rollouts in the group.
    - $f_\theta(t)$ is the importance network's prediction for the reward-variance of the rollouts at token $t$.
    - $R_1$ and $R_2$ are the rewards of the two paired rollouts.
3. Apply a soft-mask to the GRPO loss to upweight tokens with high reward-variance using the importance network (in some as-of-yet undetermined way). Number go up?


## Part 1: Initial Rollout Generation

The script `generate_paired_rollouts.py` generates a ton of rollouts using a model on a task and saves them to a directory named `rollout_dump/task_name/model_name/`.

```bash
python generate_paired_rollouts.py \
    --model_name <model_name> \
    --task_name <task_name> \
    --group_size <group_size>
```

Since we're planning to use these completions for training the importance network, we need to store the completions and rewards in a format that's easy to load and use, so we'll use a `rollouts_{args}.parquet` file with columns:

- `prompt_id`
- `prompt_text`
- `group_id` (a single prompt may be called on multiple times to generate different groups of rollouts)
- `completion_id` (i.e. 1 up to $G$)
- `split_token_idx` (the index of the token at which the rollout and its sister rollout were split)
- `completion_text`
- `reward`
- `reward_group_mean`
- `reward_group_var`
- `partial_reward_1`
- `partial_reward_2`
- `partial_reward_1_group_mean`
- `partial_reward_1_group_var`
- `partial_reward_2_group_mean`
- `partial_reward_2_group_var`
- &c.

(N.B. there's redundancy here across rows but that'll make it easier to lead the data later on when training the importance network.)

The list of partial rewards are saved in `rewards.py` as a dictionary of reward functions.

The `{args}` in `rollouts_{args}.parquet` are the arguments passed to `generate_paired_rollouts.py`, or could actually just be the group size, really.

The script `generate_paired_rollouts.py` will also save a `rollouts_metadata.json` file with the arguments passed to the script plus runtime, date stats &c.

## Part 2: Training the Importance Network

## Part 3: Applying the Soft-Mask

