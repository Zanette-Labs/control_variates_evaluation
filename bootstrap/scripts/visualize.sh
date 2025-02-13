#!/bin/bash
main_root="$HOME/projects/misc/bootstrap"
save_root="results"
main="visualize_shift_include.py"
data_list=("chatbot-arena" "mt-bench")
model_list=("skywork" "skywork" "gemma" "gemma" "armorm" "gpt4")
type_list=("finetuned" "pretrained" "finetuned" "pretrained" "pretrained" "pretrained")

for ((id=0; id<12; id++)); do
    data_id=$((id / 6))
    model_id=$((id % 6))

    data=${data_list[$data_id]}
    model=${model_list[$model_id]}
    type=${type_list[$model_id]}

    ckpt="err_dict_threshold_100_cv4_rm_human.pkl"

    python $main \
    --result_root ${save_root} \
    --dataset ${data} \
    --model ${model} \
    --model_type ${type} \
    --threshold 100 \
    --ckpt_name ${ckpt}
done