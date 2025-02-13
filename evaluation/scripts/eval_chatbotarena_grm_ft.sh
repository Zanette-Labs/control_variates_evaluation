#!/bin/bash
main="eval_rm_main.py"
config="chatbotarena_grm_ft.yaml"

for ((id=0; id<20; id++)); do
    # Find ckpt
    ckpt_matches=(../finetune/results/chatbot_arena_GRM-Gemma-2B-sftreg/run_holdout-${id}*)
    # check whether there is exactly one match
    if [ ${#ckpt_matches[@]} -eq 1 ] && [ -e "${ckpt_matches[0]}" ]; then
        ckpt="${ckpt_matches[0]}"
        echo "Checkpoint: $ckpt"
        python src/rlaihf/mains/$main test --config configs/$config --data.holdout_model_id $id \
        --model.load_weight_dir "$ckpt" \
        --trainer.devices 1 \
        --data.num_workers 1 \
        --data.keep_tie True

    else
        echo "Error: Expected exactly one match, found ${#ckpt_matches[@]}"
    fi
done