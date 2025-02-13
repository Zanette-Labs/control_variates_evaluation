# Chatbot arena datamodule for openai api

import lightning as L
from torch.utils.data import DataLoader, Dataset
import transformers
import torch
import datasets
from typing import List, Dict, Optional, Callable
import os
import numpy as np

from rlaihf.data_conversion import InverseDataConversion

class ApiDatamodule(L.LightningDataModule):
    '''
    No processing to the dataset. Batch size is fixed to 1
    '''
    def __init__(self, load_dataset_name: str, save_dataset_dir: str, load_dataset_split: Optional[str] = None, load_dataset_dir: Optional[str] = None, num_test_data: Optional[int] = None, random_sample: Optional[bool] = True, sample_seed: int = 123, data_convert: Optional[Callable] = None):
        '''
        TODO: clean the comment
        load_dataset_name: e.g., "Anthropic/hh-rlhf"
        load_dataset_dir: e.g., "harmless-base"
        save_dataset_dir: Directory to save the tokenized dataset.
        random_sample: if True, randomly sample test_data from dataset
        '''
        super().__init__()
        self.save_hyperparameters(ignore = ["_class_path"])

        self.num_test_data = num_test_data
        self.random_sample = random_sample
        self.sample_seed = sample_seed


        self.load_dataset_name = load_dataset_name
        self.load_dataset_dir = load_dataset_dir
        self.load_dataset_split = load_dataset_split
        self.save_dataset_dir = save_dataset_dir

        self._get_save_dataset_path()

        self.data_convert = data_convert

    def prepare_data(self):
        '''
        Load, tokenize, save to disk.
        '''
        test_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split=self.load_dataset_split)
        # print(self.tokenize_pipeline)
        test_dataset = test_dataset.map(self.data_convert, batched=False)
        test_dataset.save_to_disk(self.test_save_dataset_path)

    def setup(self, stage: str):
        '''
        Split dataset
        '''
        if stage == "test":
            test_dataset_full = datasets.load_from_disk(self.test_save_dataset_path)
            # Add id column to the dataset
            ids = np.arange(len(test_dataset_full))
            test_dataset_full = test_dataset_full.add_column("id", ids)
            test_dataset_full = self._filter(test_dataset_full)
            if self.num_test_data is None:
                self.test_dataset = test_dataset_full
            elif self.random_sample:
                np.random.seed(self.sample_seed)
                idxs = np.random.choice(len(test_dataset_full), size=(self.num_test_data), replace=False)
                self.test_dataset = test_dataset_full.select(idxs)
            else:
                idxs = np.arange(self.num_test_data)
                self.test_dataset = test_dataset_full.select(idxs)

    def test_dataloader(self):
        # return DataLoader(self.test_dataset,  batch_size=1, num_workers=self.num_workers)
        return self.test_dataset

    
    def _get_save_dataset_path(self):
        load_dataset_name_modified = self.load_dataset_name.replace("/", "-")
        if self.load_dataset_dir is not None:
            load_dataset_dir_modified = self.load_dataset_dir.replace("/", "-")
            load_dataset_dir_modified = f"_{load_dataset_dir_modified}"
        else:
            load_dataset_dir_modified = ""
        if self.load_dataset_split is not None:
            load_dataset_split_modified = self.load_dataset_split.replace("/", "-")
            load_dataset_split_modified = f"_{load_dataset_split_modified}"
        else:
            load_dataset_split_modified = ""

        train_save_dataset_filename = f"{load_dataset_name_modified}{load_dataset_dir_modified}{load_dataset_split_modified}_api_train.hf"
        test_save_dataset_filename = f"{load_dataset_name_modified}{load_dataset_dir_modified}{load_dataset_split_modified}_api_test.hf"
        val_save_dataset_filename = f"{load_dataset_name_modified}{load_dataset_dir_modified}{load_dataset_split_modified}_api_val.hf"

        self.train_save_dataset_path = os.path.join(self.save_dataset_dir, train_save_dataset_filename)
        self.test_save_dataset_path = os.path.join(self.save_dataset_dir, test_save_dataset_filename)
        self.val_save_dataset_path = os.path.join(self.save_dataset_dir, val_save_dataset_filename)
    
    def _filter(self, dataset):
        '''
        Filter the dataset based on some criteria
        '''
        return dataset

class ChabotArenaApiDatamodule(ApiDatamodule):
    def __init__(self, save_dataset_dir: str, num_test_data: Optional[int] = None, random_sample: Optional[bool] = True, sample_seed: int = 123):
        super().__init__(
            load_dataset_name = "lmsys/chatbot_arena_conversations", 
            save_dataset_dir = save_dataset_dir, 
            load_dataset_split = "train", 
            num_test_data = num_test_data, 
            random_sample = True, 
            sample_seed = sample_seed, 
            data_convert = InverseDataConversion(
                dialog_keys = ["conversation_a", "conversation_b"]))

    def _filter(self, dataset):
        # dataset = dataset.filter(lambda x: x["turn"] == 1 and x["language"] == "English")
        # print(f"After filtering, dataset length is {len(dataset)}")
        return dataset