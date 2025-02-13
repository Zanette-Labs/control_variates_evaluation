#!/bin/bash
data_root="../evaluation/results"
main="stats_gpt4.py"
run="mt-bench_gpt4"

# Find ckpt=
ckpt="${data_root}/$run/converted_dict.pkl"

echo $ckpt
python $main \
--rm_dict_file_path $ckpt \
--save_dir pretrained \
--save_base_dir results/$run \
--vmin 0.0 \
--vmax 0.5 \
--overwrite