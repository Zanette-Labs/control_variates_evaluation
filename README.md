# Control Variates Evaluation
Code repository for paper "Accelerating Unbiased LLM Evaluation via Synthetic Feedback"

## Run Experiments
The experiments are divided into 4 parts, corresponding to the 4 directories. Please
replicate our result in the following order:

1. (Optional) Synthetic evaluator finetuning. You can skip if you run Control Variates Evaluation with an off-the-shelf evaluator. See instructions under `finetune/`.
2. Collect Synthetic Evaluations. See instructions under `evaluation/`.
3. Compute averaged human annotation saving ratio. See instructions under `stats/`.
4. Run control variates evaluation to visualize variance and bias. See instructions under `bootstrap/`.

## Acknowledgement
Code associated with GPT-4 evaluation is partially based on [lm-sys/FastChat](https://github.com/lm-sys/FastChat).