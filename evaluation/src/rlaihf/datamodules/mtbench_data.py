from .test_data import TestDatamodule
from typing import Callable, Optional, List
import datasets
from torch.utils.data import DataLoader
import pandas as pd

def process_mtbench_dataset(dataset):
    # Average labels with same id and model pairs
    df = dataset.to_pandas()

    # Group by 'id', 'model_a', 'model_b' and aggregate 'winner' into a list
    grouped_df = df.groupby(['question_id', 'model_a', 'model_b', 'turn'])
    grouped_df = grouped_df.agg({
    'score': 'mean',
    'conversation_a': 'first',
    'conversation_b': 'first',
    })
    new_dataset = datasets.Dataset.from_pandas(grouped_df)
    return new_dataset

class MtbenchTestDatamodule(TestDatamodule):
    def prepare_data(self):
        '''
        Load, tokenize, save to disk.
        '''
        test_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split=self.load_dataset_split)
        test_dataset = test_dataset.map(self.tokenize_pipeline, batched=False)
        test_dataset = process_mtbench_dataset(test_dataset)
        indices = list(range(len(test_dataset)))
        self.test_dataset_full = test_dataset.add_column("id", indices)
        print(f"Length of dataset: {len(self.test_dataset_full)}")

class MtbenchCorrEvalDatamodule(TestDatamodule):
    def __init__(self, load_dataset_name: str, tokenize_pipeline: Callable, collate_fn: Callable, holdout_model_id: int, batch_size: int, save_dataset_dir: str, load_dataset_split: Optional[str] = None, load_dataset_dir: Optional[str] = None, num_workers: int = 0, keep_tie: bool=True):
        super().__init__(
            load_dataset_name=load_dataset_name,
            tokenize_pipeline=tokenize_pipeline,
            collate_fn=collate_fn,
            batch_size=batch_size,
            save_dataset_dir=save_dataset_dir,
            load_dataset_split=load_dataset_split,
            load_dataset_dir=load_dataset_dir,
            num_workers=num_workers
        )
        self.holdout_model_id=holdout_model_id
        self.keep_tie = keep_tie
    def prepare_data(self):
        '''
        Load, tokenize, save to disk.
        '''
        test_dataset = datasets.load_dataset(self.load_dataset_name, data_dir = self.load_dataset_dir, split=self.load_dataset_split)
        test_dataset = test_dataset.map(self.tokenize_pipeline, batched=False)
        test_dataset = process_mtbench_dataset(test_dataset)
        indices = list(range(len(test_dataset)))
        self.test_dataset_full = test_dataset.add_column("id", indices)
        print(f"Length of dataset: {len(self.test_dataset_full)}")

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            # dataset_full = datasets.load_from_disk(self.train_save_dataset_path)
            dataset_full = self.test_dataset_full
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
            self.test_dataset = dataset_full.filter(is_val)
            self.gt_scores_full = dataset_full["score"] # All scores, use index to select
        else:
            raise NotImplementedError("Only fit stage is supported.")
   