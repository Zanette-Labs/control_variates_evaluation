#!/bin/bash
# Change accordingly
id=0 # chatbot-arena_skywork
# id=1 # chatbot-arena_gemma
# id=2 # chatbot-arena_armorm
# id=3 # mt-bench_skywork
# id=4 # mt-bench_gemma
# id=5 # mt-bench_armorm

main_root="."
data_root="../evaluation/results"
save_root="results"
main="run_rm_pretrained.py"
data_list=("chatbot-arena" "mt-bench")
model_list=("skywork" "gemma" "armorm")

data_id=$((id / 3))
model_id=$((id % 3))

data=${data_list[$data_id]}
model=${model_list[$model_id]}

# Find ckpt
ckpt_matches=(${data_root}/${data}_${model}/reward_dict_pretrained.pkl)
# check whether there is exactly one match
if [ ${#ckpt_matches[@]} -eq 1 ] && [ -e "${ckpt_matches[0]}" ]; then
    ckpt="$ckpt ${ckpt_matches[0]}"
    echo "Matched file: ${ckpt_matches[0]}"
else
    echo "Error: Expected exactly one match, found ${#ckpt_matches[@]}"
    echo "Found files: ${ckpt_matches[@]}"
    # exit 1
fi

echo $ckpt
python $HOME/projects/misc/bootstrap/$main \
--rm_dict_file_path $ckpt \
--save_dir pretrained \
--save_base_dir ${save_root}/${data}_${model}\
--num_corr_samples 100 \
--overwrite

