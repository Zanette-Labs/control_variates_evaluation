import transformers
from typing import List, Dict
import torch

def get_dict_padding_collator(tokenizer: transformers.AutoTokenizer):
    '''
    Return a data collator that pads data with dict format. 
    It will
    '''
    base_data_collator = transformers.DataCollatorWithPadding(tokenizer = tokenizer, padding=True)
    def dict_collator_fn(preprocess_batch):
        def reformat_batch(batch: List[Dict[str,List[List]]]):
            new_batch = {}
            for k in batch[0]:
                new_v = []
                for data in batch:
                    new_v = new_v + data[k]
                new_batch[k] = new_v
            return new_batch
        # print(hhh_data)
        chosen_batch = [data['chosen'] for data in preprocess_batch]
        # List of dicts with keys 'attention_mask' and 'input_ids', value of type List[List], shape 1*x
        chosen_batch = reformat_batch(chosen_batch)
        chosen_batch = base_data_collator(chosen_batch)

        # print(chosen_data)
        rejected_batch = [data['rejected'] for data in preprocess_batch]
        rejected_batch = reformat_batch(rejected_batch)
        rejected_batch = base_data_collator(rejected_batch)
        return {'chosen': chosen_batch, 'rejected': rejected_batch}
    return dict_collator_fn

class StandardPaddingCollator:
    def __init__(self, model_name_or_path: str):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.base_data_collator = transformers.DataCollatorWithPadding(tokenizer = tokenizer, padding=True)
    def __call__(self, preprocess_batch):
        def reformat_batch(batch: List[Dict[str,List[List]]]):
            new_batch = {}
            for k in batch[0]:
                new_v = []
                for data in batch:
                    new_v = new_v + data[k]
                new_batch[k] = new_v
            return new_batch
        # print(hhh_data)
        chosen_batch = [data['chosen'] for data in preprocess_batch]
        # List of dicts with keys 'attention_mask' and 'input_ids', value of type List[List], shape 1*x
        chosen_batch = reformat_batch(chosen_batch)
        chosen_batch = self.base_data_collator(chosen_batch)

        # print(chosen_data)
        rejected_batch = [data['rejected'] for data in preprocess_batch]
        rejected_batch = reformat_batch(rejected_batch)
        rejected_batch = self.base_data_collator(rejected_batch)
        return {'chosen': chosen_batch, 'rejected': rejected_batch}

class StandardPaddingCollatorV2:
    def __init__(self, model_name_or_path: str, keys_to_pad: List[str]):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.base_data_collator = transformers.DataCollatorWithPadding(tokenizer = tokenizer, padding=True)
        self.keys_to_pad = keys_to_pad
    def __call__(self, preprocess_batch):
        def reformat_batch(batch: List[Dict[str,List[List]]]):
            new_batch = {}
            for k in batch[0]:
                new_v = []
                for data in batch:
                    new_v = new_v + data[k]
                new_batch[k] = new_v
            return new_batch

        pad_dict = {}

        for k in self.keys_to_pad:
            batch = [data[k] for data in preprocess_batch]
            # List of dicts with keys 'attention_mask' and 'input_ids', value of type List[List], shape 1*x
            batch = reformat_batch(batch)
            batch = self.base_data_collator(batch)
            pad_dict[k] = batch
        return pad_dict

class StandardPaddingCollatorV3:
    '''
    Can include keys that do not need padding, the values of these keys will be concatenated into a list.
    '''
    def __init__(self, model_name_or_path: str, keys_to_pad: List[str], other_keys: List[str]):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.base_data_collator = transformers.DataCollatorWithPadding(tokenizer = tokenizer, padding=True)
        # self.default_data_collator = transformers.DefaultDataCollator()
        self.keys_to_pad = keys_to_pad
        self.other_keys = other_keys

    def __call__(self, preprocess_batch):
        def reformat_batch(batch: List[Dict[str,List[List]]]):
            new_batch = {}
            for k in batch[0]:
                new_v = []
                for data in batch:
                    new_v = new_v + data[k]
                new_batch[k] = new_v
            return new_batch

        pad_dict = {}

        for k in self.keys_to_pad:
            batch = [data[k] for data in preprocess_batch]
            # List of dicts with keys 'attention_mask' and 'input_ids', value of type List[List], shape 1*x
            batch = reformat_batch(batch)
            batch = self.base_data_collator(batch)
            pad_dict[k] = batch

        batch = [{k: data[k] for k in self.other_keys} for data in preprocess_batch]
        batch = self._default_data_collator(batch)
        for k, v in batch.items():
            pad_dict[k] = v
        return pad_dict

    def _default_data_collator(self, batch_list):
        processed_batch = {}
        for k in batch_list[0].keys():
            processed_batch[k] = [b[k] for b in batch_list]
        return processed_batch
