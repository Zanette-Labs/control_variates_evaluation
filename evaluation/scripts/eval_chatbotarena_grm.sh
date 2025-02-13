#!/bin/bash
main="eval_rm_main.py"
config=chatbotarena_grm.yaml
python src/rlaihf/mains/$main test --config configs/$config
