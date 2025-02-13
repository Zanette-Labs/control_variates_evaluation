#!/bin/bash
data_root="../evaluation/results"
main="stats_finetuned_rm.py"
run_list=("chatbot-arena_skywork" "chatbot-arena_gemma")

for id in 0 1; do
    run=${run_list[$id]} 
    lr=${lr_list[$id]}
    # Find ckpt
    ckpt=""
    for ((id=0; id<20; id++)); do
        ckpt_matches=(${data_root}/${run}/reward_dict_run_holdout-${id}_*.pkl)
        # check whether there is exactly one match
        if [ ${#ckpt_matches[@]} -eq 1 ] && [ -e "${ckpt_matches[0]}" ]; then
            ckpt="$ckpt ${ckpt_matches[0]}"
            echo "Matched file: ${ckpt_matches[0]}"
        else
            echo "Error: Expected exactly one match, found ${ckpt_matches[@]}"
        fi
    done

    python $main \
    --rm_dict_file_path $ckpt \
    --save_dir finetuned \
    --save_base_dir results/$run \
    --vmin 0.0 \
    --vmax 0.9 \
    --overwrite
done