partition=cnu
code=$PROJ_CODE

gpu_type="rtx_3090"
num_gpus=1
mem=62

NUM_CPU=1
TIME=24

group_sizes=(16 32)
tasks=(gsm8k countdown)

lora_ranks=(8 16)


for task in ${tasks[@]}; do
    for group_size in ${group_sizes[@]}; do
        job_name="_impnet_${task}_${group_size}_fft"
        lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $job_name --conda-env grpo \
            --cmd "python train_importance_network.py --rollout_model_name Qwen/Qwen3-0.6B --rollout_task_name $task --group_size $group_size --num_epochs 1"

        for lora_rank in ${lora_ranks[@]}; do
            job_name="_impnet_${task}_${group_size}_lora${lora_rank}"
            lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $job_name --conda-env grpo \
                --cmd "python train_importance_network.py --rollout_model_name Qwen/Qwen3-0.6B --rollout_task_name $task --group_size $group_size --num_epochs 1 --finetune_type lora --lora_rank $lora_rank"
        done
    done
done