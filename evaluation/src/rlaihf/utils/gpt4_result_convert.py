from typing import List, Tuple, Dict, Optional
import numpy as np
from copy import deepcopy
from collections import defaultdict
import sys, os
import json
import pickle
import re

def winner2score(winner_list: Tuple[str, List[str]]) -> Tuple[float, List[float]]:
    def winner2score_single(winner: str) -> float:
        if winner == "model_a":
            return 1.0
        elif winner == "model_b":
            return 0.0
        elif "tie" in winner:
            return 0.5
        else:
            raise ValueError(f"Unknown winner: {winner}")
    if isinstance(winner_list, str):
        return winner2score_single(winner_list)
    return [winner2score_single(winner) for winner in winner_list]


class ApiDataConversion:
    def __init__(self, original_dataset, data_file_dir: str, data_file_name: str = "annotated_data_all.json"):
        self.original_dataset = original_dataset
        self.data_file_dir = data_file_dir
        self.data_file_name = data_file_name
        self.data = self.load_data()
    def load_data(self):
        data_file_path = os.path.join(self.data_file_dir, self.data_file_name)
        print(f"Loading data from {data_file_path}")
        with open(data_file_path, "r") as f:
            data = json.load(f)
        return data
    def _parse_api_pref(self, datum: dict):
        completion = datum["api_completions"]

        regex = re.compile("\[\[(A|B|C)\]\]")
        match = regex.search(completion)
        if match is None:
            print(f"Warning: Cannot find preference")
            return None
        pref = match.group()
        assert pref[:2] == "[[", pref
        assert pref[-2:] == "]]", pref
        
        pref = pref.replace("[", "")
        pref = pref.replace("]", "")

        if pref == "A":
            return 1.0
        if pref == "B":
            return 0.0
        if pref == "C":
            return 0.5
        raise NotImplementedError(f"{datum}")

    
    def convert_single_dict(self, datum: dict):
        api_score = self._parse_api_pref(datum)
        if api_score is None:
            return None
        original_data = self.original_dataset[datum["id"]]
        return {
            "api_score": api_score,
            "gt_score": winner2score(datum["winner"]),
            "model_a": original_data["model_a"],
            "model_b": original_data["model_b"],
            "id": datum["id"]
        }
    def convert_and_save(self):
        converted_data = []
        for datum in self.data:
            converted_datum = self.convert_single_dict(datum)
            if converted_datum is not None:
                converted_data.append(converted_datum)
        print(f"Get {len(converted_data)} valid data, {len(self.data)} in total")
        final_data_dict = {}
        for k in converted_data[0].keys():
            final_data_dict[k] = [d[k] for d in converted_data]
            final_data_dict[k] = np.asarray(final_data_dict[k])
        save_path = os.path.join(self.data_file_dir, "converted_dict.pkl")
        if os.path.exists(save_path):
            print(f"Warning: {save_path} already exists. Skip the saving process")
        else:
            with open(save_path, "wb") as f:
                pickle.dump(final_data_dict, f)
            print(f"Converted data saved to {save_path}")
        return final_data_dict


