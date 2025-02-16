# fix the generating models
# Use ArmoRM as automatic annotator
# Update: reward dict contains model names and ground truth

import torch
import numpy as np
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from collections import defaultdict
import utils
import argparse
import os, json
import sys
from typing import List

def load_rm_rewards_v2(file_path: List[str]):
    '''
    Do not concatenate, return a list of np.ndarray
    '''
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
# def load_gt_data(dataset_path: str, split: str):
#     '''
#     Load ground truth scores and model names
#     '''
#     gt_data = datasets.load_dataset(dataset_path, split=split)
#     # gt_data = gt_data.remove_columns(["prompt", "response"])
#     # gt_scores = [
#     #     gt_data
#     # ]
#     gt_data = pd.DataFrame(gt_data)
#     model_a_names = gt_data["model_a"]
#     model_b_names = gt_data["model_b"]
#     winners = gt_data["winner"]
#     return model_a_names.to_numpy(), model_b_names.to_numpy(), winners.tolist()

def winner2score(winner: str) -> float:
    if 'tie' in winner:
        return 0.5
    if winner == 'model_a':
        return 1.0
    if winner == 'model_b':
        return 0.0
    raise NotImplementedError
    
def get_gt_notie_idx(gt_scores):
    return gt_scores != 0.5

def aggregate(model_a_names, model_b_names, rm_scores, gt_scores):
    rm_dict = defaultdict(list)
    gt_dict = defaultdict(list)
    for model_a, model_b, rm, gt in zip(model_a_names, model_b_names, rm_scores, gt_scores):
        rm_dict[(model_a, model_b)].append(rm)
        gt_dict[(model_a, model_b)].append(gt)
    return rm_dict, gt_dict

def get_agreement(rm_scores, gt_scores):
    rm_scores = np.asarray(rm_scores)
    gt_scores = np.asarray(gt_scores)
    return 1 - np.mean(np.abs(rm_scores - gt_scores))

def get_winrate(gt_scores):
    return np.mean(gt_scores)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rm_dict_file_path', type=str, nargs='+', default="/home/zhaoyiz/projects/misc/rm_finetune/results/chatbot-arena_gemma/reward_dict_run_holdout-0_lr-1e-6_batch-32_notie_checkpoint-600.pkl")
    parser.add_argument('--save_base_dir', type=str, default="results/chatbot-arena_gemma")
    parser.add_argument('--save_dir', type=str, default="None")
    parser.add_argument('--keep_ties', action="store_false")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--vmin', type=float, default=0.0)
    parser.add_argument('--vmax', type=float, default=1.0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # save args
    if args.save_dir != "None":
        dir_name = args.save_dir
    elif len(args.rm_dict_file_path) == 1:
        ckpt_name = args.rm_dict_file_path[0].split("/")[-1].split(".")[0]
        dir_name = ckpt_name
    else:
        print("Warning: save_dir is not specified")
        dir_name = "multi_files_default"
    if not args.keep_ties:
        dir_name += "_notie"
    save_dir = os.path.join(args.save_base_dir, dir_name)
    os.makedirs(save_dir, exist_ok=args.overwrite)
    print("Saving results to ", save_dir)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # config output logging
    sys.stdout = utils.DualLogger(os.path.join(save_dir, "output.log"))
    # clear log
    with open(os.path.join(save_dir, "output.log"), "w") as f:
        pass

    # Load data
    conv_a_reward, conv_b_reward, model_a_names, model_b_names, gt_scores = load_rm_rewards_v2(args.rm_dict_file_path)
    rm_scores = [utils.get_rm_score_bt(ra, rb) for ra, rb in zip(conv_a_reward, conv_b_reward)]

    # process data
    grouper = utils.ResultGrouperV2(
        model_a_arr = model_a_names,
        model_b_arr = model_b_names,
        rm_scores = rm_scores,
        gt_scores = gt_scores,
        keep_ties = args.keep_ties
    )


    # Fix model correlation
    result_dict = grouper.get_corr_mat()
    sorted_model_list = result_dict["list"]

    utils.visualize_matrix(result_dict["corr"], sorted_model_list,   
      file_path = os.path.join(save_dir, "corr_matrix.png"),
      kwargs_dict = {
          "show_diagonal": False,
          "nan_label": "N/A",
          "vmin": args.vmin,
          "vmax": args.vmax
      })
    utils.visualize_matrix(result_dict["winrate"], sorted_model_list,   
      file_path = os.path.join(save_dir, "winrate_matrix.png"))
    utils.visualize_matrix(result_dict["acc"], sorted_model_list,   
      file_path = os.path.join(save_dir, "accuracy_matrix.png"))
    utils.visualize_matrix(result_dict["num"], sorted_model_list,   
      file_path = os.path.join(save_dir, "num_matrix.png"),
      kwargs_dict = {"type": "int"})
    utils.visualize_matrix(result_dict["l1"], sorted_model_list,   
        file_path = os.path.join(save_dir, "rm_l1_loss.png"),
        kwargs_dict = {
            "show_diagonal": False,
            "nan_label": "N/A",
            "vmin": 0.0,
            "vamx": 1.0
        })
    utils.visualize_matrix(result_dict["l2"], sorted_model_list,
        file_path = os.path.join(save_dir, "rm_l2_loss.png"),
        kwargs_dict = {
            "show_diagonal": False,
            "nan_label": "N/A",
            "vmin": 0.0,
            "vamx": 1.0
        })
    utils.visualize_matrix(result_dict["diff"], sorted_model_list,
        file_path = os.path.join(save_dir, "rm_winrate_diff.png"),
        kwargs_dict = {
            "show_diagonal": False,
            "nan_label": "N/A",
            "vmin": 0.0,
            "vamx": 1.0
        })
    if "mt-bench" in args.rm_dict_file_path[0]:
        saving_vmax = 20
    else:
        saving_vmax = 50
    utils.visualize_matrix(result_dict["saving"], sorted_model_list,
        file_path = os.path.join(save_dir, "saving.png"),
        kwargs_dict = {
            "show_diagonal": False,
            "nan_label": "N/A",
            "type": "int",
            "vmin": 0,
            "vmax": saving_vmax
        })

    # Compute average saving
    for threshold in [0, 50, 100, 150, 200]:
        utils.compute_sample_saving(threshold, result_dict["corr"], result_dict["num"])
        print()




