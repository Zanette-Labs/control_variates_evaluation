#!/bin/bash
main="run_rm_ft.py"
# Change the following arguments accordingly
data=chatbot-arena # or mt-bench
n=20 # or 6 for mt-bench
model=skywork # or gemma

run=${data}_${model}

# Find ckpt
ckpt=""
for ((id=0; id<$n; id++)); do
    ckpt_matches=(../evaluation/results/${run}/reward_dict_run_holdout-${id}_*.pkl)
    # check whether there is exactly one match
    if [ ${#ckpt_matches[@]} -eq 1 ] && [ -e "${ckpt_matches[0]}" ]; then
        ckpt="$ckpt ${ckpt_matches[0]}"
    else
        echo "Error: Expected exactly one match, found ${ckpt_matches[@]}"
    fi
done

echo $ckpt
python $main \
--rm_dict_file_path $ckpt \
--save_dir finetuned \
--save_base_dir results/$run \
--num_corr_samples 100 \
--overwrite
