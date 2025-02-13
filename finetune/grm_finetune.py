# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
trl==0.11.0
transformers==4.40.0

grm finetune
"""

from utils import prepare_dataset

import warnings

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
import os

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    setup_chat_format,
)
from trl.commands.cli_utils import RewardScriptArguments
from trl.extras.dataset_formatting import conversations_formatting_function
from dataclasses import dataclass, field
import wandb
import numpy as np
from typing import Dict, Union, Any, Tuple, Optional, List

class GemmaRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        _, _, rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )
        _, _, rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )
        rewards_chosen = rewards_chosen.unsqueeze(-1) # (B,1)
        rewards_rejected = rewards_rejected.unsqueeze(-1) # (B,1)
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

@dataclass
class CustomArguments(RewardConfig):
    holdout_model_id: int = field(
        default=0,
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
    parser = HfArgumentParser((RewardScriptArguments, CustomArguments, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_model = model_config.model_name_or_path.split("/")[-1]
    config.output_dir = os.path.join("results", f"{args.dataset_name}_{current_model}", config.log_base_dir)

    if is_main_process():
        wandb.init(
            project="RM-finetune", 
            name=f"{args.dataset_name}_{current_model}_{config.log_base_dir}"
        )

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        use_cache=False if config.gradient_checkpointing else True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    #############################
    # Load and preprocess dataset
    #############################

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the chosen/rejected columns are in the OpenAI messages format.
        chosen_fn = conversations_formatting_function(tokenizer, "chosen")
        rejected_fn = conversations_formatting_function(tokenizer, "rejected")
        train_dataset, eval_dataset = prepare_dataset(dataset_name = args.dataset_name, holdout_model_id = config.holdout_model_id, keep_tie = config.keep_tie)
        train_dataset = train_dataset.map(
            lambda x: {"chosen": chosen_fn(x), "rejected": rejected_fn(x)}, num_proc=config.dataset_num_proc
        )
        eval_dataset = eval_dataset.map(
            lambda x: {"chosen": chosen_fn(x), "rejected": rejected_fn(x)}, num_proc=config.dataset_num_proc
        )

        # Tokenize inputs
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
        # Filter out examples that are too long
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )

    ##########
    # Training
    ##########
    trainer = GemmaRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    metrics = trainer.evaluate()
    if is_main_process():
        trainer.save_model(config.output_dir)
        print(f"Save model to {config.output_dir}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.save_model(config.output_dir)
        print(f"Model hold out id {config.holdout_model_id}")
        print("Finished!")