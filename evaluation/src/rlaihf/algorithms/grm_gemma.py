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
from safetensors import safe_open

class GrmGemma(L.LightningModule):
    def __init__(self, save_dir: str, dialog_keys: List[str], model_name_or_path: str = "Ray2333/GRM-Gemma-2B-sftreg", model_dtype: str = "float16", load_weight_dir: Optional[str] = None): 
        super().__init__()
        self.save_hyperparameters(ignore = ["_class_path"])
        
        self.model_dtype = torch_dtype_mapping(model_dtype)
        self.backbone_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, torch_dtype=self.model_dtype, trust_remote_code = True)
        # load finetuned weight if needed
        if load_weight_dir is not None:
            print(f"Loading weight from {load_weight_dir}")
            self._load_weight(load_weight_dir)
        self.load_weight_dir = load_weight_dir
        self.backbone_model.eval()

        self.save_dir = save_dir

        self.dialog_keys = dialog_keys
        self.reward_dict = defaultdict(list)

    def _load_weight(self, load_weight_dir: str):

        state_dict = {}
            
        # Open the shard file
        shard_file_path = os.path.join(load_weight_dir, "model.safetensors")
        with safe_open(shard_file_path, framework="pt") as f:
            # Iterate over all tensor keys in the shard
            for key in f.keys():
                # print(key)
                # Get the tensor and store it in the state_dict
                if "v_head" in key:
                    new_key = key
                else:
                    new_key = f"pretrained_model.{key}"
                state_dict[new_key] = f.get_tensor(key)
        # Load the state_dict into the model
        missing_keys, unexpected_keys = self.backbone_model.load_state_dict(state_dict, strict=False)

        # Print any missing or unexpected keys
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    def forward(self, tokenized_batch):
        _, _, reward_tensor = self.backbone_model(**tokenized_batch)
        return reward_tensor

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            for key in self.dialog_keys:
                reward_tensor = self.forward(batch[key])                
                self.reward_dict[f"{key}_reward"].append(reward_tensor.cpu())

        ids = batch["id"]
        model_a = batch["model_a"]
        model_b = batch["model_b"]
        gt_score = batch["score"]
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
        os.makedirs(save_dir, exist_ok=True)

        # Get save path postfix
        postfix = ""
        if self.load_weight_dir is not None:
            paths = self.load_weight_dir.split("/")
            for p in paths:
                if 'run' in p or 'checkpoint' in p:
                    postfix = f"{postfix}_{p}"
        if postfix == "":
            postfix = "_pretrained"
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
