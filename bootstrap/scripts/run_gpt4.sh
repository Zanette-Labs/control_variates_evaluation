#!/bin/bash
# Change accordingly
id=1 # chatbot-arena_gpt4, or 1 for mt-bench_gpt4s

main_root="."
data_root="../evaluation/results"
save_root="results"
main="run_gpt4.py"
data_list=("chatbot-arena" "mt-bench")
data=${data_list[$id]}
model=gpt4

# Find ckpt
ckpt_matches=(${data_root}/${data}_${model}*/converted_dict.pkl)
# check whether there is exactly one match
if [ ${#ckpt_matches[@]} -eq 1 ] && [ -e "${ckpt_matches[0]}" ]; then
    ckpt="$ckpt ${ckpt_matches[0]}"
    echo "Matched file: ${ckpt_matches[0]}"
else
    echo "Error: Expected exactly one match, found ${#ckpt_matches[@]}"
    echo "Found files: ${ckpt_matches[@]}"
fi

echo $ckpt
python $HOME/projects/misc/bootstrap/$main \
--rm_dict_file_path $ckpt \
--save_dir pretrained \
--save_base_dir ${save_root}/${data}_${model} \
--num_corr_samples 100 \
--overwrite

