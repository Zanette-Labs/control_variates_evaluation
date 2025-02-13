# Control Variates Evaluation - Bootstrap

Change directory to `bootstrap/`. Then run
```bash
bash scripts/run_gpt4.sh
bash scripts/run_rm_ft.sh
bash scripts/run_rm_pretrained.sh
```
For each script, you need to change the variable `id` manually to complete all runs. See the comments in the bash script.

Every script uses CPU only and takes 10 minutes to a few hours depending on the number of CPUs you have. It will automatically detect the number of CPUs you allocate and run multi-processing.

After the bootstrap step, you can visualize the variance and error by running
```bash
bash scripts/visualize.sh
```
The script assumes that you have run all 12 bootstrap experiments in the previous steps.