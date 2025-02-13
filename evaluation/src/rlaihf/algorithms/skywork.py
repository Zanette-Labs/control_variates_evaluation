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

class SkyworkLlama(L.LightningModule):
    def __init__(self, save_dir: str, dialog_keys: List[str], label_key: str, model_name_or_path: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", model_dtype: str = "bfloat16", lr: float = 1e-3, weight_decay: float = 0., load_weight_dir: Optional[str] = None): 
        super().__init__()
        self.save_hyperparameters(ignore = ["_class_path"])
        
        self.model_dtype = torch_dtype_mapping(model_dtype)

        self.backbone_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, torch_dtype=model_dtype, trust_remote_code=True)
        if load_weight_dir is not None:
            print(f"Loading weight from {load_weight_dir}")
            self._load_weight(load_weight_dir)
        self.load_weight_dir = load_weight_dir
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.backbone_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.backbone_model.train()
        # self.backbone_model = None
        # self.model_name_or_path = model_name_or_path

        self.save_dir = save_dir
        self.label_key = label_key
        self.dialog_keys = dialog_keys
        self.reward_dict = defaultdict(list)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.val_losses = []
        self.val_preds = []
        self.val_ids = []

    def _load_weight(self, load_weight_dir: str):
        index_file = os.path.join(load_weight_dir, 'model.safetensors.index.json')

        # Check if the index file exists
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found at {index_file}")

        # Load the index data
        with open(index_file, 'r') as f:
            index_data = json.load(f)

        # Extract the mapping from parameters to shard filenames
        param_to_filename = index_data['weight_map']

        # Get the unique shard filenames
        shard_filenames = set(param_to_filename.values())

        state_dict = {}

        # Iterate over each shard file
        for shard_filename in shard_filenames:
            shard_file_path = os.path.join(load_weight_dir, shard_filename)
                
            # Open the shard file
            with safe_open(shard_file_path, framework="pt") as f:
                # Iterate over all tensor keys in the shard
                for key in f.keys():
                    # Skywork model does not need to change key names
                    new_key = f"{key}"
                    state_dict[new_key] = f.get_tensor(key)
        # Load the state_dict into the model
        missing_keys, unexpected_keys = self.backbone_model.load_state_dict(state_dict, strict=False)

        # Print any missing or unexpected keys
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    def forward(self, tokenized_batch):
        output = self.backbone_model(**tokenized_batch)
        score = output.logits.reshape(-1)

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone_model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        return {
            "optimizer": optimizer,
        }

    def training_step(self, batch, batch_idx):
        scores = []
        for key in self.dialog_keys:
            score = self.forward(batch[key])
            scores.append(score)
        pred = scores[0] - scores[1]
        label = batch[self.label_key]
        label = label.to(pred.device)
        loss = self.loss_fn(pred, label)
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        scores = []
        for key in self.dialog_keys:
            score = self.forward(batch[key])
            scores.append(score)
        pred = scores[0] - scores[1]
        label = batch[self.label_key]
        label = label.to(pred.device)
        loss = self.loss_fn(pred, label)
        self.val_losses.append(loss.cpu())
        self.val_preds.append(pred.cpu())

        ids = batch['id']
        self.val_ids += ids

    def on_validation_epoch_end(self):
        all_losses = self.all_gather(self.val_losses)[0] # tensor(xx, xx, ...)
        self.val_losses.clear()  # free memory
        all_preds = self.all_gather(self.val_preds) # list of tensor, shape num_device * batch
        self.val_preds.clear()
        all_ids = self.all_gather(self.val_ids)
        self.val_ids.clear()

        all_preds = torch.cat(all_preds, dim=0).squeeze(dim=-1) # (batch * num_device)
        all_preds = all_preds.cpu().type(torch.float32).numpy()
        all_ids = torch.cat(all_ids, dim=0)
        all_ids = all_ids.cpu().numpy()

        mean = torch.mean(all_losses)

        val_gt_score = self.gt_scores_full[all_ids]
        corr = np.corrcoef(val_gt_score, all_preds)[0,1]

        if self.trainer.is_global_zero:
            self.log("val_loss", mean, rank_zero_only=True)
            self.log("val_corr", corr, rank_zero_only=True)
            self.log("val_num_data", len(all_ids), rank_zero_only=True)

    def on_fit_start(self):
        self.gt_scores_full = np.asarray(self.trainer.datamodule.gt_scores_full)

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
        file_name = f"reward_dict{postfix}.pkl"
        with open(os.path.join(save_dir, file_name), "wb") as f:
            pickle.dump(final_dict, f)
        print(f"Save reward dict to {os.path.join(save_dir, file_name)}")

        # compute correlation
        rewards = []
        for key in self.dialog_keys:
            rewards.append(final_dict[f"{key}_reward"])
        assert len(rewards) == 2
        pred = rewards[0] - rewards[1]
        gt_scores_test = final_dict["gt_score"]
        corr = np.corrcoef(gt_scores_test, pred)[0,1]
        print(f"Correlation: {corr}")