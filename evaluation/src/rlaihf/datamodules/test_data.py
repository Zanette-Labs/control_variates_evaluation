import lightning as L
from torch.utils.data import DataLoader, Dataset
import transformers
import torch
import datasets
from typing import List, Dict, Optional, Callable
import os
import numpy as np

class TestDatamodule(L.LightningDataModule):
    def __init__(self, load_dataset_name: str, tokenize_pipeline: Callable, collate_fn: Callable, batch_size: int, save_dataset_dir: str, load_dataset_split: Optional[str] = None, load_dataset_dir: Optional[str] = None, num_workers: int = 0, num_test_data: Optional[int] = None):
        '''
        TODO: clean the comment
        load_dataset_name: e.g., "Anthropic/hh-rlhf"
        load_dataset_dir: e.g., "harmless-base"
        save_dataset_dir: Directory to save the tokenized dataset.
        '''
        super().__init__()
        self.save_hyperparameters()
        self.batch_size     = batch_size

        self.num_workers = num_workers
        self.num_test_data = num_test_data

        self.tokenize_pipeline = tokenize_pipeline
        self.collate_fn = collate_fn

        self.load_dataset_name = load_dataset_name
        self.load_dataset_dir = load_dataset_dir
        self.load_dataset_split = load_dataset_split

        self.save_dataset_dir = save_dataset_dir

        self._get_save_dataset_path()


    def prepare_data(self):
        '''
        Load, tokenize, save to disk.
        '''
        # if not os.path.exists(self.train_save_dataset_path):
        #     train_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split="train")           
        #     train_dataset = train_dataset.map(self.tokenize_pipeline, batched=False)
        #     train_dataset.save_to_disk(self.train_save_dataset_path)
     
        # if not os.path.exists(self.val_save_dataset_path):
        #     val_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split="test")
        #     val_dataset = val_dataset.map(self.tokenize_pipeline, batched=False)
        #     val_dataset.save_to_disk(self.val_save_dataset_path)

        # if not os.path.exists(os.path.expanduser(self.test_save_dataset_path)):
        
        # This process should be done in every run.
        # First, this is not time-consuming. 
        # Second, if we use different model, tokenization changes, so the tokenized data are different.
        test_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split=self.load_dataset_split)
        # print(self.tokenize_pipeline)
        test_dataset = test_dataset.map(self.tokenize_pipeline, batched=False)
        indices = list(range(len(test_dataset)))
        self.test_dataset_full = test_dataset.add_column("id", indices)
        # test_dataset.save_to_disk(self.test_save_dataset_path)

    def setup(self, stage: str):
        '''
        Split dataset
        '''
        # if stage == "fit":
        #     train_dataset_full = datasets.load_from_disk(self.train_save_dataset_path)
        #     val_dataset_full = datasets.load_from_disk(self.val_save_dataset_path)
        #     if self.num_train_data is None:
        #         self.train_dataset = train_dataset_full
        #     else:
        #         idxs = np.random.choice(len(train_dataset_full), size=(self.num_train_data), replace=False)
        #         self.train_dataset = train_dataset_full.select(idxs)

        #     if self.num_val_data is None:
        #         self.val_dataset = val_dataset_full
        #     else:
        #         idxs = np.random.choice(len(val_dataset_full), size=(self.num_val_data), replace=False)
        #         self.val_dataset = val_dataset_full.select(idxs)
        if stage == "test":
            # test_dataset_full = datasets.load_from_disk(self.test_save_dataset_path)
            test_dataset_full = self.test_dataset_full
            if self.num_test_data is None:
                self.test_dataset = test_dataset_full
            else:
                idxs = np.random.choice(len(test_dataset_full), size=(self.num_test_data), replace=False)
                self.test_dataset = test_dataset_full.select(idxs)

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn = self.collate_fn)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset,  batch_size=self.batch_size, num_workers=self.num_workers, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  batch_size=self.batch_size, num_workers=self.num_workers, collate_fn = self.collate_fn)

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

        train_save_dataset_filename = f"{load_dataset_name_modified}{load_dataset_dir_modified}{load_dataset_split_modified}_train.hf"
        test_save_dataset_filename = f"{load_dataset_name_modified}{load_dataset_dir_modified}{load_dataset_split_modified}_test.hf"
        val_save_dataset_filename = f"{load_dataset_name_modified}{load_dataset_dir_modified}{load_dataset_split_modified}_val.hf"

        self.train_save_dataset_path = os.path.join(self.save_dataset_dir, train_save_dataset_filename)
        self.test_save_dataset_path = os.path.join(self.save_dataset_dir, test_save_dataset_filename)
        self.val_save_dataset_path = os.path.join(self.save_dataset_dir, val_save_dataset_filename)

# if __name__ == '__main__':
#     datamodule = HelpfulHarmlessGemma(2,4,10,10,"Ray2333/GRM-Gemma-2B-sftreg")
#     # print(datamodule.batch_size)
#     datamodule.prepare_data()
#     # datamodule.train_dataloader()
#     print(datamodule.train_dataset[0])
#     # for data in datamodule.train_dataset:
#     #     print(len(data['chosen']))

    