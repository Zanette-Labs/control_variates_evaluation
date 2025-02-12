# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
skywork finetuning
"""

from prepare_dataset import prepare_dataset

import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from datetime import datetime
import os


from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    # get_kbit_device_map,
    # get_peft_config,
    # get_quantization_config,
    setup_chat_format,
)
# from trl.commands.cli_utils import RewardScriptArguments
from dataclasses import dataclass, field
import wandb 
import numpy as np

@dataclass
class CustomArguments(RewardConfig):
    # custom_arg: str = field(
    #     default="default_value",
    #     metadata={"help": "Description of the custom argument"}
    # )
    # dataset_name: str = field(
    #     default="chatbot_arena",
    #     # metadata={"help": "Another custom argument"}
    # )
    holdout_model_id: int = field(
        default=0,
        # metadata={"help": "Another custom argument"}
    )
    log_base_dir: str = field(
        default="default",
    )
    keep_tie: bool = field(
        default=False,
    )

def is_main_process():
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    return local_rank == 0


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, CustomArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    current_model = model_args.model_name_or_path.split("/")[-1]
    training_args.output_dir = os.path.join("results", f"{script_args.dataset_name}_{current_model}", training_args.log_base_dir)

    if is_main_process():
        wandb.init(
            project="RM-finetune", 
            name=f"{script_args.dataset_name}_{current_model}_{training_args.log_base_dir}"
        )

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
    #     warnings.warn(
    #         "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
    #         " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
    #         UserWarning,
    #     )

    ##############
    # Load dataset
    ##############
    # print(script_args)
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset, eval_dataset = prepare_dataset(script_args.dataset_name, holdout_model_id = training_args.holdout_model_id, keep_tie=training_args.keep_tie)
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Eval dataset: {len(eval_dataset)} examples")
    # print(train_dataset)

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
        # peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    # metrics = trainer.evaluate()
    # if is_main_process():
        # trainer.save_model(training_args.output_dir)
        # print(f"Save model to {config.output_dir}")
        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
        # trainer.save_model(training_args.output_dir)
        # trainer.push_to_hub()
        # print(f"Model hold out id {training_args.holdout_model_id}")
    print("Finished!")
    # trainer.save_model(training_args.output_dir)

    # if training_args.eval_strategy != "no":
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)