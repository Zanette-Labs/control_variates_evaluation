import lightning as L
from fastchat.llm_judge.common import load_judge_prompts
from fastchat.model.model_adapter import get_conversation_template
from openai import OpenAI
import json
import os
from typing import List
import numpy as np
from pathlib import Path

class OpenaiApi(L.LightningModule):
    def __init__(self, save_dir: str, judge_file_path: str, prompt_type: str, record_keys: List[str], model_name: str = "gpt-4", ckpt_period: int = 64, from_batch: int = 0): 
        '''
        record_keys: keys in original dataset to record
        '''
        super().__init__()
        self.save_hyperparameters(ignore = ["_class_path"])
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.record_keys = record_keys
        self.ckpt_period = ckpt_period
        self.from_batch = from_batch

        judge_prompts = load_judge_prompts(judge_file_path)
        self.judge_prompt = judge_prompts[prompt_type]


        self.client = OpenAI()

        self.annotated_data = []
        self.annotated_data_ckpt = []
        self.cur_batch_idx = 0

    def test_step(self, batch, batch_idx):
        '''
        Batch size is always 1
        '''
        if batch_idx < self.from_batch:
            return
        system_prompt = self.judge_prompt["system_prompt"]
        user_prompt = self.judge_prompt["prompt_template"].format(
            question=batch["prompt"],
            answer_a=batch["response_1"],
            answer_b=batch["response_2"],
        )
        conv = get_conversation_template(self.model_name)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        conv.set_system_message(system_prompt)
        messages = conv.to_openai_api_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )
        output = response.choices[0].message.content

        output_data = {k: batch[k] for k in self.record_keys}
        output_data["api_completions"] = output
        self.annotated_data.append(output_data)
        self.annotated_data_ckpt.append(output_data)

        if (batch_idx + 1) % self.ckpt_period == 0: # save ckpt, and clear record
            ckpt = (batch_idx + 1) // self.ckpt_period
            self._save_data(self.annotated_data_ckpt, f"annotated_data_ckpt_{ckpt}")
            self.annotated_data_ckpt = []

        self.cur_batch_idx = batch_idx

    def on_test_end(self):
        if len(self.annotated_data_ckpt) > 0: # save the rest of the data
            ckpt = self.cur_batch_idx // self.ckpt_period + 1
            self._save_data(self.annotated_data_ckpt, f"annotated_data_ckpt_{ckpt}")

    def _save_data(self, data, file_name: str):
        file_path = os.path.join(self.save_dir, file_name)
        if Path(file_path).exists():
            raise ValueError(f"File {file_path} already exists")
        with open(file_path, "w") as f:
            json.dump(data, f, indent = 2)

    def _load_data(self, file_name: str):
        with open(os.path.join(self.save_dir, file_name)) as f:
            data = json.load(f)
        return data