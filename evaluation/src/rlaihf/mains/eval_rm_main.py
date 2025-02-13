import lightning as L
from lightning.pytorch.cli import LightningCLI

import torch
import argparse
import transformers

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--seed", type=int, default=0)
        parser.link_arguments("model.init_args.model_name_or_path", "data.init_args.tokenize_pipeline.init_args.model_name_or_path")
        parser.link_arguments("model.init_args.model_name_or_path", "data.init_args.collate_fn.init_args.model_name_or_path")
        parser.link_arguments("model.init_args.dialog_keys", "data.init_args.tokenize_pipeline.init_args.dialog_keys")
        parser.link_arguments("model.init_args.dialog_keys", "data.init_args.collate_fn.init_args.keys_to_pad")
    def before_test(self):
        L.seed_everything(self.config['test']['seed'], workers=True)
    def before_fit(self):
        L.seed_everything(self.config['fit']['seed'], workers=True)
    def after_fit(self):
        self.trainer.logger.info("Finished training!")
        self.trainer.print(torch.cuda.memory_summary())


def cli_main():
    cli = MyLightningCLI(save_config_callback=None)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    cli_main()
    # main()
    # print("Finished")