partition=cnu
code=$PROJ_CODE

gpu_type="rtx_3090"
num_gpus=1
mem=62

NUM_CPU=1
TIME=24

group_sizes=(16 32 64)

for group_size in ${group_sizes[@]}; do
    lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n rollouts_$group_size --conda-env grpo \
        --cmd "python generate_paired_rollouts.py --model_name Qwen/Qwen3-0.6B --task_name gsm8k --group_size $group_size --num_epochs 1 --output_dir rollout_dump"
done