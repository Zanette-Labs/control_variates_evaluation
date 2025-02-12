#!/bin/bash

# Hyperparameters
batch=32 # global batch size
lr=1e-6
n_gpus=2 # number of gpus used. We use 2 H100 GPUs. Change accordingly
device_batch=8 # batch size on each gpu

echo "Batch size: $batch, Learning rate: $lr"

# Runtime configs
for ((id=0; id<20; id++)); do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    base_dir="run_holdout-${id}_lr-${lr}_batch-${batch}_notie_${timestamp}"

    n_proc=${n_gpus}
    grad_accum=$((batch/device_batch/n_proc))

    # Argument output_dir is a placeholder, it will be overwritten in the python script.
    accelerate launch --num_processes ${n_proc} --zero_stage 2 --main_process_port=$((29500+id)) \
    grm_finetune.py \
    --dataset_name chatbot_arena \
    --holdout_model_id ${id} \
    --log_base_dir ${base_dir} \
    --keep_tie False \
    --evaluation_strategy 'steps' \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --model_name_or_path 'Ray2333/GRM-Gemma-2B-sftreg' \
    --output_dir results/GRM-Gemma-2B-Reward \
    --torch_dtype float16 \
    --per_device_train_batch_size ${device_batch} \
    --per_device_eval_batch_size ${device_batch} \
    --gradient_accumulation_steps ${grad_accum} \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --trust_remote_code True \
    --learning_rate $lr \
    --logging_steps 10 \
    --bf16 True\
    --remove_unused_columns False \
    --max_length 2048 \
    --dataloader_num_workers ${SLURM_CPUS_PER_TASK} \
done