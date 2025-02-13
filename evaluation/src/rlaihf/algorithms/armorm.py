
from rlaihf.utils import torch_dtype_mapping
import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from copy import deepcopy
import os, json
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download
from typing import Optional, List
import pickle
from collections import defaultdict
import numpy as np

class ArmoRM(L.LightningModule):
    def __init__(self, save_dir: str, dialog_keys: List[str], model_name_or_path: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1", model_dtype: str = "bfloat16"): 
        super().__init__()
        self.save_hyperparameters(ignore = ["_class_path"])
        
        self.model_dtype = torch_dtype_mapping(model_dtype)
        self.backbone_model = None
        self.model_name_or_path = model_name_or_path

        self.save_dir = save_dir

        self.dialog_keys = dialog_keys
        self.reward_dict = defaultdict(list)
       
    def configure_model(self):
        if self.backbone_model is not None:
            return
        self.backbone_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, torch_dtype=self.model_dtype, trust_remote_code=True, ignore_mismatched_sizes=True)
        self.backbone_model.train()

    def forward(self, tokenized_batch):
        output = self.backbone_model(**tokenized_batch)
        multi_obj_rewards = output.rewards.cpu().float()
        preference_score = output.score.cpu().float()  

        return multi_obj_rewards, preference_score

    def test_step(self, batch, batch_idx):
        '''
        Test win rate
        '''
        with torch.no_grad():
            for key in self.dialog_keys:
                multi_obj_rewards, preference_score = self.forward(batch[key])
                self.reward_dict[f"{key}_multi_obj_rewards"].append(multi_obj_rewards)
                self.reward_dict[f"{key}_reward"].append(preference_score)

        ids = batch["id"]
        model_a = batch["model_a"]
        model_b = batch["model_b"]
        gt_score = batch["score"]
        # print("step id", ids)
        self.reward_dict["id"] += ids
        self.reward_dict["model_a"] += model_a
        self.reward_dict["model_b"] += model_b
        self.reward_dict["gt_score"] += gt_score

    def on_test_end(self):
        final_dict = {}
        for key, values in self.reward_dict.items():
            if isinstance(values[0], torch.Tensor):
                values = torch.cat(values, dim=0)
                final_dict[key] = values.numpy()
            else:
                final_dict[key] = np.asarray(values)

        save_dir = self.save_dir
        postfix = "_pretrained"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"reward_dict{postfix}.pkl"), "wb") as f:
            pickle.dump(final_dict, f)

        # compute correlation
        rewards = []
        for key in self.dialog_keys:
            rewards.append(final_dict[f"{key}_reward"])
        assert len(rewards) == 2
        pred = rewards[0] - rewards[1]
        gt_scores_test = final_dict["gt_score"]
        corr = np.corrcoef(gt_scores_test, pred)[0,1]
        print(f"Correlation: {corr}")

        # Acc
        pred_scores = (rewards[0] > rewards[1]).astype(float)
        def get_accuracy(pred, gt):
            if gt == 0.5:
                return 0.5
            else:
                return float(gt == pred)
        acc_list = [get_accuracy(pred, gt) for pred, gt in zip(pred_scores, gt_scores_test)]
        print("Acc", np.mean(acc_list))