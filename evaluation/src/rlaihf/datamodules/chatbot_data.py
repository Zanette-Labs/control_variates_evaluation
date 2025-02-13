import lightning as L
from torch.utils.data import DataLoader, Dataset
import transformers
import torch
import datasets
from typing import List, Dict, Optional, Callable
import os
import numpy as np
from datetime import datetime

class ChatBotArenaCorrEvalDatamodule(L.LightningDataModule):
    def __init__(self, load_dataset_name: str, tokenize_pipeline: Callable, collate_fn: Callable, holdout_model_id: int, batch_size: int, save_dataset_dir: str, load_dataset_split: Optional[str] = None, load_dataset_dir: Optional[str] = None, num_workers: int = 0, keep_tie: bool=True):
        super().__init__()
        self.save_hyperparameters(ignore = ["tokenize_pipeline", "collate_fn", "_class_path"])
        self.batch_size     = batch_size
        self.num_workers = num_workers
        self.tokenize_pipeline = tokenize_pipeline
        self.collate_fn = collate_fn
        self.load_dataset_name = load_dataset_name
        self.load_dataset_dir = load_dataset_dir
        self.load_dataset_split = load_dataset_split
        self.save_dataset_dir = save_dataset_dir
        self.holdout_model_id = holdout_model_id
        self.keep_tie = keep_tie

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            # dataset_full = datasets.load_from_disk(self.train_save_dataset_path)
            dataset_full = self.train_dataset_full
            model_a = dataset_full['model_a']
            model_b = dataset_full['model_b']
            self.model_set = set(model_a + model_b)
            model_list = sorted(list(self.model_set))
            print(f"{len(self.model_set)} different models")
            assert self.holdout_model_id < len(self.model_set)
            holdout_model = model_list[self.holdout_model_id]
            print(f"Evaluation holdout model: {holdout_model}")
            def is_train(datum) -> bool:
                if not self.keep_tie:
                    tie_flag = ('tie' not in datum['winner'])
                else:
                    tie_flag = True
                return datum['model_a'] != holdout_model and datum['model_b'] != holdout_model and tie_flag
                
            def is_val(datum) -> bool:
                if not self.keep_tie:
                    tie_flag = ('tie' not in datum['winner'])
                else:
                    tie_flag = True
                return tie_flag and (datum['model_a'] == holdout_model or datum['model_b'] == holdout_model)
            self.train_dataset = dataset_full.filter(is_train)
            self.val_dataset = dataset_full.filter(is_val)
            self.gt_scores_full = dataset_full["score"] # All scores, use index to select
        else:
            raise NotImplementedError("Only fit stage is supported.")
            
    def prepare_data(self):
        '''
        Added a column "id" to the dataset.
        '''
        
        train_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split=self.load_dataset_split)
        train_dataset = train_dataset.map(self.tokenize_pipeline, batched=False)
        indices = list(range(len(train_dataset)))
        self.train_dataset_full = train_dataset.add_column("id", indices)

    def test_dataloader(self):
        print("Eval dataset size", len(self.val_dataset))
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)