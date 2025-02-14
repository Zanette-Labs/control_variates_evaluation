#!/bin/bash
data_root="../evaluation/results"
main="stats_pretrained_rm.py"
run_list=("chatbot-arena_skywork" "chatbot-arena_gemma" "chatbot-arena_armorm")

for id in 0 1 2; do
    run=${run_list[$id]}
    # Find ckpt
    ckpt="${data_root}/${run}/reward_dict_pretrained.pkl"

    echo $ckpt
    python $main \
    --rm_dict_file_path $ckpt \
    --save_dir pretrained \
    --save_base_dir results/$run \
    --vmin 0.0 \
    --vmax 0.9 \
    --overwrite
done