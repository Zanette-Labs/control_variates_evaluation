from typing import List, Dict, Tuple, Callable
from copy import deepcopy
import sys

class DictDataConversion:
    def __init__(self, prior_keys: List[List[str]], post_keys: List[str], dialog_convert: Callable):
        '''
        prior_keys: each element is a list of keys, the values of which are args to dialog_convert. The order is important, and should follow the dialog_convert parameters order.

        post_keys: Dialog keys after data conversion. Ensure len(post_keys) == len(prior_keys)

        Pipeline: Copy the data, add keys in <post_keys> (if exist in data then modify original values)
        '''
        assert len(prior_keys) == len(post_keys)
        self.prior_keys = prior_keys
        self.post_keys = post_keys
        self.dialog_convert = dialog_convert
    def __call__(self, dict_data: Dict):
        convert_data = deepcopy(dict_data)
        for key_list, post_key in zip(self.prior_keys, self.post_keys):
            raw_dialog_args = [dict_data[key] for key in key_list]
            std_dialog = self.dialog_convert(*raw_dialog_args)
            convert_data[post_key] = std_dialog
        return convert_data
