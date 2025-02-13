import lightning as L
from lightning.pytorch.cli import LightningCLI

import torch
import argparse
import transformers

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--seed", type=int, default=0)
    def before_test(self):
        print(self.config['test'])
        L.seed_everything(self.config['test']['seed'], workers=True)
    def before_fit(self):
        L.seed_everything(self.config['fit']['seed'], workers=True)


def cli_main():
    cli = MyLightningCLI(save_config_callback=None)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    cli_main()
    # main()
    print("Finished")