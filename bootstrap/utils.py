from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import sys
import pickle
from pathlib import Path
import re
import os

# def load_rm_dict(path: str) -> Dict[str, List[float]]:
#     '''
#     Load a dictionary from a file
#     '''
#     with open(path, 'rb') as f:
#         return pickle.load(f)

# def rm_gt_scores(rm_dict, use_bt_modelling: bool = True):
#     conv_a_reward = rm_dict['conversation_a_reward']
#     conv_b_reward = rm_dict['conversation_b_reward']
#     reward_diff = conv_b_reward - conv_a_reward
#     if use_bt_modelling:
#         rm_score = 1 / (1 + np.exp(reward_diff))
#     else:
#         rm_score = reward_diff
#     gt_score = rm_dict['gt_score']
#     return rm_score, gt_score

def load_rm_rewards(file_path: List[str]):
    reward_dict_list = []
    for p in file_path:
        with open(p, "rb") as f:
            reward_dict = pickle.load(f)
            reward_dict_list.append(reward_dict)

    # Do not need to scale, linear transform does not affect correlation
    # Find conv reward key name. They are not fixed across models
    for key in reward_dict_list[0].keys():
        if "conversation_a" in key:
            conv_a_key = key
        if "conversation_b" in key:
            conv_b_key = key
    print("Conv a key", conv_a_key)
    print("Conv b key", conv_b_key)
    conv_a_reward = [reward_dict[conv_a_key] for reward_dict in reward_dict_list]
    conv_b_reward = [reward_dict[conv_b_key] for reward_dict in reward_dict_list]
    model_a = [reward_dict["model_a"] for reward_dict in reward_dict_list]
    model_b = [reward_dict["model_b"] for reward_dict in reward_dict_list]
    gt_score = [reward_dict["gt_score"] for reward_dict in reward_dict_list]
    conv_a_reward = np.concatenate(conv_a_reward, axis=0)
    conv_b_reward = np.concatenate(conv_b_reward, axis=0)
    model_a = np.concatenate(model_a, axis=0)
    model_b = np.concatenate(model_b, axis=0)
    gt_score = np.concatenate(gt_score, axis=0)
    return conv_a_reward, conv_b_reward, model_a, model_b, gt_score

def load_rm_rewards_v2(file_path: List[str]):
    reward_dict_list = []
    for p in file_path:
        with open(p, "rb") as f:
            reward_dict = pickle.load(f)
            reward_dict_list.append(reward_dict)

    # Do not need to scale, linear transform does not affect correlation
    # Find conv reward key name. They are not fixed across models
    for key in reward_dict_list[0].keys():
        if "conversation_a" in key:
            conv_a_key = key
        if "conversation_b" in key:
            conv_b_key = key
    print("Conv a key", conv_a_key)
    print("Conv b key", conv_b_key)
    conv_a_reward = [reward_dict[conv_a_key] for reward_dict in reward_dict_list]
    conv_b_reward = [reward_dict[conv_b_key] for reward_dict in reward_dict_list]
    model_a = [reward_dict["model_a"] for reward_dict in reward_dict_list]
    model_b = [reward_dict["model_b"] for reward_dict in reward_dict_list]
    gt_score = [reward_dict["gt_score"] for reward_dict in reward_dict_list]
    # conv_a_reward = np.concatenate(conv_a_reward, axis=0)
    # conv_b_reward = np.concatenate(conv_b_reward, axis=0)
    # model_a = np.concatenate(model_a, axis=0)
    # model_b = np.concatenate(model_b, axis=0)
    # gt_score = np.concatenate(gt_score, axis=0)
    return conv_a_reward, conv_b_reward, model_a, model_b, gt_score

def get_rm_score(conv_a_reward, conv_b_reward, use_bt_modelling: bool = True):
    # exp_conv_a_reward = np.exp(conv_a_reward)
    # exp_conv_b_reward = np.exp(conv_b_reward)
    reward_diff = conv_b_reward - conv_a_reward
    if use_bt_modelling:
        rm_scores = 1 / (1 + np.exp(reward_diff))
    else:
        rm_scores = reward_diff
    return rm_scores

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

    
    def get_corr_mat(self):
        model_list = self.model_a_arr.tolist() + self.model_b_arr.tolist()
        model_set = set(model_list)
        sorted_model_list = sorted(list(model_set))
        print("Number of models", len(sorted_model_list))
        corr_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        num_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        win_rate_mat = 0.5 + np.zeros((len(sorted_model_list), len(sorted_model_list)))
        acc_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
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
        # if return_win_rate_mat and return_acc_mat:
        return corr_mat, num_mat, win_rate_mat, acc_mat, sorted_model_list
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

    
    def get_corr_mat(self):
        sorted_model_list = self.sorted_model_list
        print("Number of models", len(sorted_model_list))
        corr_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        num_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        win_rate_mat = 0.5 + np.zeros((len(sorted_model_list), len(sorted_model_list)))
        acc_mat = np.zeros((len(sorted_model_list), len(sorted_model_list)))
        for i, model_a in enumerate(sorted_model_list):
            for j, model_b in enumerate(sorted_model_list):
                if model_a == model_b:
                    corr_mat[i, j] = 1
                elif (model_a, model_b) in self.rm_dict.keys():
                    rm_scores = self.rm_dict[(model_a, model_b)]
                    gt_scores = self.gt_dict[(model_a, model_b)]
                    corr = np.corrcoef(rm_scores, gt_scores)[0,1]
                    corr_mat[i, j] = corr
                    # corr_mat[j, i] = corr
                    num_mat[i, j] = len(rm_scores)
                    # num_mat[j, i] = len(rm_scores)

                    win_rate_mat[i,j] = np.mean(gt_scores)
                    # win_rate_mat[j,i] = 1 - np.mean(gt_scores)

                    acc_mat[i,j] = self._get_accuracy(rm_scores, gt_scores)
                    # acc_mat[j,i] = self._get_accuracy(rm_scores, gt_scores)
        # if return_win_rate_mat and return_acc_mat:
        return corr_mat, num_mat, win_rate_mat, acc_mat, sorted_model_list
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
        flipped_model_b_names = deepcopy(model_b_names)
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

# def safe_sample_without_replacement(n, size):
#     '''
#     if size[-1] <= n, then sample each row without replacement
#     else, the first n elements are a shuffle, and sample the rest with replacement
#     '''
#     if size[-1] <= n:
#         idx_list = [np.random.choice(n, size=size[-1], replace=False) for _ in range(np.prod(size[:-1]))]
#     else:
#     idx_arr = np.concatenate(idx_list, axis=0)
        
#     else:
#         idx = np.arange(n)
#         np.random.shuffle(idx)
#         return np.concatenate([idx, np.random.choice(n, size=size[1:], replace=True)], axis=0)

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



if __name__ == '__main__':
    # Redirect stdout to a file
    rm_path = "/home/zhaoyiz/projects/misc/rm_finetune/results/chatbot-arena_gemma/reward_dict_pretrained.pkl"
    rm_dict = load_rm_dict(rm_path)
    print(rm_dict.keys())