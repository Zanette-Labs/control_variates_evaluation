from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import sys, os
import json
import pickle
import re
from matplotlib.colors import Normalize
from pathlib import Path

def get_rm_score_bt(conv_a_reward, conv_b_reward):
    reward_diff = conv_b_reward - conv_a_reward
    scores = 1 / (1 + np.exp(reward_diff))
    return scores

class ResultGrouper:
    def __init__(self, model_a_arr: np.ndarray[str], model_b_arr: np.ndarray[str], rm_scores: Tuple[List[float], np.ndarray], gt_scores: Tuple[List[float], np.ndarray], sort_model: bool, keep_ties: bool):
        self.model_a_arr = model_a_arr
        self.model_b_arr = model_b_arr

        self.rm_scores = np.asarray(rm_scores)
        self.gt_scores = np.asarray(gt_scores)
        if not keep_ties:
            self._remove_ties()
        if sort_model:
            self._sort_by_model_name()
        self.rm_dict, self.gt_dict = self.aggregate()

    def get_total_correlation(self):
        return np.corrcoef(self.rm_scores, self.gt_scores)[0,1]

    def get_total_winrate(self):
        return np.mean(self.gt_scores)

    def get_total_accuracy(self):
        acc_list = []
        for rm, gt in zip(self.rm_scores, self.gt_scores):
            if gt == 0.5:
                continue
            else:
                acc_list.append(int(rm > 0.5) == int(gt > 0.5))
        return np.mean(acc_list)

    def _get_accuracy(self, rm_scores, gt_scores):
        acc_list = []
        for rm, gt in zip(rm_scores, gt_scores):
            if gt == 0.5:
                continue
            else:
                acc_list.append(int(rm > 0.5) == int(gt > 0.5))
        return np.mean(acc_list)
    
    def _get_winrate_diff(self, rm_scores, gt_scores):
        rm_scores = np.asarray(rm_scores)
        gt_scores = np.asarray(gt_scores)
        return np.abs(np.mean(rm_scores) - np.mean(gt_scores))
    
    def _get_loss(self, rm_scores, gt_scores, ord: int):
        rm_scores = np.asarray(rm_scores)
        gt_scores = np.asarray(gt_scores)
        if ord == 1:
            return np.mean(np.abs(rm_scores - gt_scores))
        if ord == 2:
            return np.mean(np.square(rm_scores - gt_scores))
        else:
            raise NotImplementedError
    
    def get_corr_mat(self):
        model_list = self.model_a_arr.tolist() + self.model_b_arr.tolist()
        model_set = set(model_list)
        sorted_model_list = sorted(list(model_set))
        print("Number of models", len(sorted_model_list))
        corr_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        num_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        win_rate_mat = 0.5 + np.zeros((len(sorted_model_list), len(sorted_model_list)))
        acc_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        rm_l1_loss_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        rm_l2_loss_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        rm_winrate_diff_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        saving_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))

        for i, model_a in enumerate(sorted_model_list):
            for j, model_b in enumerate(sorted_model_list):
                if model_a == model_b:
                    corr_mat[i, j] = 1
                elif (model_a, model_b) in self.rm_dict.keys():
                    rm_scores = self.rm_dict[(model_a, model_b)]
                    gt_scores = self.gt_dict[(model_a, model_b)]
                    corr = np.corrcoef(rm_scores, gt_scores)[0,1]
                    corr_mat[i, j] = corr
                    corr_mat[j, i] = corr
                    num_mat[i, j] = len(rm_scores)
                    num_mat[j, i] = len(rm_scores)

                    win_rate_mat[i,j] = np.mean(gt_scores)
                    win_rate_mat[j,i] = 1 - np.mean(gt_scores)

                    acc_mat[i,j] = self._get_accuracy(rm_scores, gt_scores)
                    acc_mat[j,i] = self._get_accuracy(rm_scores, gt_scores)

                    l1_loss = self._get_loss(rm_scores, gt_scores, ord=1)
                    rm_l1_loss_mat[i,j] = l1_loss
                    rm_l1_loss_mat[j,i] = l1_loss

                    l2_loss = self._get_loss(rm_scores, gt_scores, ord=2)
                    rm_l2_loss_mat[i,j] = l2_loss
                    rm_l2_loss_mat[j,i] = l2_loss
                    
                    winrate_diff = self._get_winrate_diff(rm_scores, gt_scores)
                    rm_winrate_diff_mat[i,j] = winrate_diff
                    rm_winrate_diff_mat[j,i] = winrate_diff

                    saving_mat[i,j] = corr * corr * 100
                    saving_mat[j,i] = corr * corr * 100
        # if return_win_rate_mat and return_acc_mat:
        return {
            "corr": corr_mat,
            "num": num_mat,
            "winrate": win_rate_mat,
            "acc": acc_mat,
            "l1": rm_l1_loss_mat,
            "l2": rm_l2_loss_mat,
            "diff": rm_winrate_diff_mat, 
            "list": sorted_model_list,
            "saving": saving_mat
        }
    # corr_mat, num_mat, win_rate_mat, acc_mat, rm_l1_loss_mat, rm_l2_loss_mat, sorted_model_list
        # if return_win_rate_mat:
        #     return corr_mat, num_mat, win_rate_mat, sorted_model_list
        # if return_acc_mat:
        #     return corr_mat, num_mat, acc_mat, sorted_model_list
        # return corr_mat, num_mat, sorted_model_list

    def _sort_by_model_name(self):
        flips = self.model_a_arr > self.model_b_arr
        self.rm_scores = self._flip_scores(self.rm_scores, flips)
        self.gt_scores = self._flip_scores(self.gt_scores, flips)
        self.model_a_arr, self.model_b_arr = self._flip_model_names(self.model_a_arr, self.model_b_arr, flips)

    def _flip_scores(self, scores: np.ndarray, flips: np.ndarray):
        '''
        flips: bool array. If True, then this element in scores needs to flip
        '''
        num_data = len(scores)
        flipped_scores = deepcopy(scores)
        flipped_scores[flips] = 1 - scores[flips]
        return flipped_scores

    def _flip_model_names(self, model_a_names, model_b_names, flips):
        flipped_model_a_names = deepcopy(model_a_names)
        flipped_model_b_names = deepcopy(model_b_names)
        for i in range(len(model_a_names)):
            if flips[i]:
                flipped_model_a_names[i], flipped_model_b_names[i] = model_b_names[i], model_a_names[i]
        return flipped_model_a_names, flipped_model_b_names

    def _remove_ties(self):
        ties = self.gt_scores == 0.5
        self.rm_scores = self.rm_scores[~ties]
        self.gt_scores = self.gt_scores[~ties]
        self.model_a_arr = self.model_a_arr[~ties]
        self.model_b_arr = self.model_b_arr[~ties]

    def aggregate(self):
        rm_dict = defaultdict(list)
        gt_dict = defaultdict(list)
        for model_a, model_b, rm, gt in zip(self.model_a_arr, self.model_b_arr, self.rm_scores, self.gt_scores):
            rm_dict[(model_a, model_b)].append(rm)
            gt_dict[(model_a, model_b)].append(gt)
        return rm_dict, gt_dict

class ResultGrouperV2:
    '''
    Updated way to get matrix
    '''
    def __init__(self, model_a_arr: List[np.ndarray[str]], model_b_arr: List[np.ndarray[str]], rm_scores: List[np.ndarray[str]], gt_scores: List[np.ndarray[str]], keep_ties: bool):
        self.model_a_arr = model_a_arr
        self.model_b_arr = model_b_arr

        self.sorted_model_list = self.get_sorted_model_list()

        # self.rm_scores = np.asarray(rm_scores)
        # self.gt_scores = np.asarray(gt_scores)
        self.rm_scores = rm_scores
        self.gt_scores = gt_scores
        if not keep_ties:
            self._remove_ties()
        self._sort_by_holdout_model_name()
        self.rm_dict, self.gt_dict = self.aggregate()

    def get_sorted_model_list(self): 
        model_list = self.model_a_arr + self.model_b_arr
        model_list = np.concatenate(model_list, axis=0).tolist()
        model_set = set(model_list)
        return sorted(list(model_set))

    # def get_total_correlation(self):
    #     return np.corrcoef(self.rm_scores, self.gt_scores)[0,1]

    # def get_total_winrate(self):
    #     return np.mean(self.gt_scores)

    # def get_total_accuracy(self):
    #     acc_list = []
    #     for rm, gt in zip(self.rm_scores, self.gt_scores):
    #         if gt == 0.5:
    #             continue
    #         else:
    #             acc_list.append(int(rm > 0.5) == int(gt > 0.5))
    #     return np.mean(acc_list)

    def _get_winrate_diff(self, rm_scores, gt_scores):
        rm_scores = np.asarray(rm_scores)
        gt_scores = np.asarray(gt_scores)
        return np.abs(np.mean(rm_scores) - np.mean(gt_scores))

    def _get_accuracy(self, rm_scores, gt_scores):
        acc_list = []
        for rm, gt in zip(rm_scores, gt_scores):
            if gt == 0.5:
                continue
            else:
                acc_list.append(int(rm > 0.5) == int(gt > 0.5))
        return np.mean(acc_list)
        
    def _get_loss(self, rm_scores, gt_scores, ord: int):
        rm_scores = np.asarray(rm_scores)
        gt_scores = np.asarray(gt_scores)
        if ord == 1:
            return np.mean(np.abs(rm_scores - gt_scores))
        if ord == 2:
            return np.mean(np.square(rm_scores - gt_scores))
        else:
            raise NotImplementedError
    
    def get_corr_mat(self):
        sorted_model_list = self.sorted_model_list
        print("Number of models", len(sorted_model_list))
        corr_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        num_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        win_rate_mat = 0.5 + np.zeros((len(sorted_model_list), len(sorted_model_list)))
        acc_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        rm_l1_loss_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        rm_l2_loss_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        rm_winrate_diff_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        saving_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        for i, model_a in enumerate(sorted_model_list):
            for j, model_b in enumerate(sorted_model_list):
                if model_a == model_b:
                    corr_mat[i, j] = 1
                elif (model_a, model_b) in self.rm_dict.keys():
                    rm_scores = self.rm_dict[(model_a, model_b)]
                    gt_scores = self.gt_dict[(model_a, model_b)]
                    corr = np.corrcoef(rm_scores, gt_scores)[0,1]
                    corr_mat[i, j] = corr
                    num_mat[i, j] = len(rm_scores)
                    win_rate_mat[i,j] = np.mean(gt_scores)
                    acc_mat[i,j] = self._get_accuracy(rm_scores, gt_scores)
                    rm_l1_loss_mat[i,j] = self._get_loss(rm_scores, gt_scores, ord=1)
                    rm_l2_loss_mat[i,j] = self._get_loss(rm_scores, gt_scores, ord=2)
                    rm_winrate_diff_mat[i,j] = self._get_winrate_diff(rm_scores, gt_scores)
                    saving_mat[i,j] = corr * corr * 100
        # if return_win_rate_mat and return_acc_mat:
        return {
            "corr": corr_mat,
            "num": num_mat,
            "winrate": win_rate_mat,
            "acc": acc_mat,
            "l1": rm_l1_loss_mat,
            "l2": rm_l2_loss_mat,
            "diff": rm_winrate_diff_mat, 
            "list": sorted_model_list,
            "saving": saving_mat
        }
        # if return_win_rate_mat:
        #     return corr_mat, num_mat, win_rate_mat, sorted_model_list
        # if return_acc_mat:
        #     return corr_mat, num_mat, acc_mat, sorted_model_list
        # return corr_mat, num_mat, sorted_model_list

    def _sort_by_holdout_model_name(self):
        sorted_model_a_arr = []
        sorted_model_b_arr = []
        sorted_rm_scores = []
        sorted_gt_scores = []
        for holdout_model, model_a, model_b, rm, gt in zip(self.sorted_model_list, self.model_a_arr, self.model_b_arr, self.rm_scores, self.gt_scores):
            model_a_is_holdout = model_a == holdout_model
            model_b_is_holdout = model_b == holdout_model
            # print(model_a_is_holdout, model_b_is_holdout)
            assert np.all(model_a_is_holdout == ~model_b_is_holdout)
            flips = model_b_is_holdout
            rm = self._flip_scores(rm, flips)
            gt = self._flip_scores(gt, flips)
            model_a, model_b = self._flip_model_names(model_a, model_b, flips)
            sorted_model_a_arr.append(model_a)
            sorted_model_b_arr.append(model_b)
            sorted_rm_scores.append(rm)
            sorted_gt_scores.append(gt)
        self.model_a_arr = np.concatenate(sorted_model_a_arr,axis=0)
        self.model_b_arr = np.concatenate(sorted_model_b_arr,axis=0)
        self.rm_scores = np.concatenate(sorted_rm_scores,axis=0)
        self.gt_scores = np.concatenate(sorted_gt_scores,axis=0)
    def _flip_scores(self, scores: np.ndarray, flips: np.ndarray):
        '''
        flips: bool array. If True, then this element in scores needs to flip
        '''
        num_data = len(scores)
        flipped_scores = deepcopy(scores)
        flipped_scores[flips] = 1 - scores[flips]
        return flipped_scores

    def _flip_model_names(self, model_a_names, model_b_names, flips):
        flipped_model_a_names = deepcopy(model_a_names)
        flipped_model_a_names = flipped_model_a_names.astype(object)
        flipped_model_b_names = deepcopy(model_b_names)
        flipped_model_b_names = flipped_model_b_names.astype(object)
        for i in range(len(model_a_names)):
            if flips[i]:
                flipped_model_a_names[i], flipped_model_b_names[i] = model_b_names[i], model_a_names[i]
        return flipped_model_a_names, flipped_model_b_names

    def _remove_ties(self):
        ties = [gt == 0.5 for gt in self.gt_scores]
        self.rm_scores = [rm[~ties] for rm in self.rm_scores]
        self.gt_scores = [gt[~ties] for gt in self.gt_scores]
        self.model_a_arr = [arr[~ties] for arr in self.model_a_arr]
        self.model_b_arr = [arr[~ties] for arr in self.model_b_arr]

    def aggregate(self):
        rm_dict = defaultdict(list)
        gt_dict = defaultdict(list)
        for model_a, model_b, rm, gt in zip(self.model_a_arr, self.model_b_arr, self.rm_scores, self.gt_scores):
            rm_dict[(model_a, model_b)].append(rm)
            gt_dict[(model_a, model_b)].append(gt)
        return rm_dict, gt_dict

def get_corr_mat_from_dict(sorted_model_list, rm_dict, gt_dict):
    corr_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
    num_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
    std_prod_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
    cov_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
    win_rate_mat = 0.5 + np.zeros((len(sorted_model_list), len(sorted_model_list)))
    for i, model_a in enumerate(sorted_model_list):
        for j, model_b in enumerate(sorted_model_list):
            if model_a == model_b:
                corr_mat[i, j] = 1
            elif (model_a, model_b) in rm_dict.keys():
                rm_scores = rm_dict[(model_a, model_b)]
                gt_scores = gt_dict[(model_a, model_b)]
                corr = np.corrcoef(rm_scores, gt_scores)[0,1]
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
                num_mat[i, j] = len(rm_scores)
                num_mat[j, i] = len(rm_scores)

                cov = np.cov(rm_scores, gt_scores)[0,1]
                cov_mat[i, j] = cov
                cov_mat[j, i] = cov

                std_prod = np.std(rm_scores) * np.std(gt_scores)
                std_prod_mat[i, j] = std_prod
                std_prod_mat[j, i] = std_prod

                win_rate_mat[i,j] = np.mean(gt_scores)
                win_rate_mat[j,i] = 1 - np.mean(gt_scores)
    return corr_mat, num_mat, std_prod_mat, cov_mat, win_rate_mat

def visualize_matrix(mat: np.ndarray, row_col_labels: List[str], file_path: str, kwargs_dict: Optional[Dict] = None): 
    '''
    kwargs_dict:
        - "type": "int"/"float"
        - "show_diagonal": True/False. Default True
        - "nan_label": str
        - "vmin"
        - "vmax"
    '''
    # Define default parameters:
    default_kwargs_dict = {
        "type": "float",
        "show_diagonal": True,
        "nan_label": "nan",
        "vmin": None,
        "vmax": None
    }

    if kwargs_dict is not None:
        for k, v in kwargs_dict.items():
            default_kwargs_dict[k] = v
        

    # Set the figure size
    n = len(row_col_labels)
    plt.figure(figsize=(1 + n * 4, 1 + n * 4))

    if not default_kwargs_dict["show_diagonal"]:
        np.fill_diagonal(mat, np.nan)

    if default_kwargs_dict["vmin"] is not None and default_kwargs_dict["vmax"] is not None:
        norm = Normalize(vmin = default_kwargs_dict["vmin"], vmax = default_kwargs_dict["vmax"])
        plt.matshow(mat, cmap="viridis", norm=norm)
    else:
        plt.matshow(mat, cmap="viridis")
    # round_mat = np.round(mat, 3)
    def process(x, nan_label = "nan"):
        if np.isnan(x):
            return nan_label
        if default_kwargs_dict["type"] == "int":
            return f"{int(x)}"
        if x == 1:
            return "1"
        return f"{x:.2f}".lstrip('0')
    for (i,j), num in np.ndenumerate(mat):
        plt.text(j,i, process(num, nan_label=default_kwargs_dict["nan_label"]), ha='center', va='center',c='r', fontsize=6)
    plt.xticks(np.arange(n), row_col_labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(n), row_col_labels, fontsize=8)

    # colorbar
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

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

def compute_sample_saving(threshold, corr_mat, num_mat):
    '''
    corr_mat might contain nan
    '''
    # threshold = 100
    saving_mat = corr_mat * corr_mat
    # Remove nan and pairs with small number of data points
    num_mask = np.logical_and(~np.isnan(corr_mat), num_mat > threshold)
    masked_num_mat = num_mat * num_mask
    avg_saving = np.nansum(saving_mat * num_mask) / np.sum(num_mask)
    weighted_avg_saving = np.nansum(saving_mat * masked_num_mat) / np.sum(masked_num_mat)
    print(f"Threshold {threshold}")
    print(f"{np.sum(num_mask)} model pairs have more than {threshold} data points with non-nan correlation")
    print(f"Average saving: {avg_saving}")
    print(f"Weighted average saving: {weighted_avg_saving}")

    # print out non-zero numbers
    # id = np.nonzero(masked_num_mat)
    # print(f"Number: {masked_num_mat[id]}")
    # print(f"Saving: {saving_mat[id]}")

class DualLogger:
    '''
    Save printed output to a file
    '''
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")  # Append mode

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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

def find_single_file_by_regex(directory, pattern):
    """
    Finds a single file in the specified directory that matches the given regex pattern.

    :param directory: Path to the directory to search in.
    :param pattern: Regex pattern to match the filename.
    :return: Path object of the matching file.
    :raises FileNotFoundError: If no file matches the pattern.
    :raises Exception: If multiple files match the pattern.
    """
    # Compile the regex pattern for efficiency
    regex = re.compile(pattern)
    
    # Create a Path object for the directory
    path = Path(directory)
    
    # Ensure the directory exists
    if not path.is_dir():
        raise NotADirectoryError(f"The path {directory} is not a valid directory.")
    
    # Iterate over all files in the directory and find matches
    matching_files = [f for f in path.iterdir() if f.is_file() and regex.match(f.name)]
    
    if len(matching_files) == 1:
        return matching_files[0]
    elif not matching_files:
        raise FileNotFoundError(f"No file found matching the pattern: {pattern}")
    else:
        raise Exception(f"Multiple files found matching the pattern: {pattern}")

def load_pickle_data(file_path: str, ambiguous=True):
    file_path_split = file_path.split("/")
    directory = "/".join(file_path_split[:-1])
    filename = file_path_split[-1]
    if ambiguous:
        file_path = find_single_file_by_regex(directory, filename)
    else:
        file_path = os.path.join(directory, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

