# Control Variates Evaluation - Finetune
This part is optional. If you use a pretrained synthetic evaluator, directly go to `evaluation/`.


## Run Experiments
Change directory to `finetune/`. Then run
```bash
bash grm_chatbotarena_finetune.sh
bash grm_mtbench_finetune.sh
bash skywork_chatbotarena_finetune.sh
bash skywork_mtbench_finetune.sh
```

Note: When running scripts with names including `grm`, use the following package versions to ensure the code runs correctly:
```txt
trl==0.11.0
transformers==4.40.0
```