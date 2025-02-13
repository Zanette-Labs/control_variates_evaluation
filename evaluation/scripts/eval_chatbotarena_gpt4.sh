#!/bin/bash
main="eval_gpt4_main.py"
config="chatbotarena_gpt4.yaml"

python src/rlaihf/mains/$main test --config configs/$config

# Convert to standard format for unified stats
python src/rlaihf/mains/convert_chatbotarena_gpt4.py
