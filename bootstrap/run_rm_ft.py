from utils import load_rm_rewards_v2, get_rm_score, DualLogger, ResultGrouperV2
from evaluators import HumanOnlyEvaluator, RmOnlyEvaluator, ControlVariatesEvaluator, ControlVariatesEvaluatorV2, ControlVariatesEvaluatorV3, ControlVariatesEvaluatorV4

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
import multiprocessing

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rm_dict_file_path', type=str, nargs='+', required=True)
    parser.add_argument('--save_base_dir', type=str, default="results/chatbot-arena_skywork")
    parser.add_argument('--save_dir', type=str, default="None")
    parser.add_argument('--sort_model', action="store_false")
    parser.add_argument('--keep_ties', action="store_false")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--bt', action="store_false", help="Use BT modelling")
    parser.add_argument('--num_corr_samples', type=int, default=100, help="Number of samples to estimate the correlation between the ground truth scores and the reward modelling scores.")
    parser.add_argument('--threshold', type=int, default=100, help="Threshold to filter pairs")
    parser.add_argument('--repeat', type=int, default=1000, help="Number of repetitions for the bootstrap")
    args = parser.parse_args()
    return args

# define evaluation process
def evaluate_pair(model_pair, eval_type, evaluator, num_samples, gt_winrate):
    evaluator.clear_result()
    print(f"Pair {model_pair}")
    for num_gt_samples in range(10, 2001, 1):
        evaluator.evaluate_batch(num_gt_samples, args.repeat)

    return model_pair, {
        'num_samples': num_samples,
        'pred_type': eval_type,
        'prediction': evaluator.get_eval_result(),
        'gt_winrate': gt_winrate
    }

if __name__ == "__main__":
    args = get_args()
    assert args.threshold >= args.num_corr_samples, "Threshold should be no less than num_corr_samples"

    # save args
    if len(args.rm_dict_file_path) == 1:
        ckpt_name = args.rm_dict_file_path[0].split("/")[-1].split(".")[0]
        dir_name = ckpt_name
    elif args.save_dir != "None":
        dir_name = args.save_dir
    else:
        print("Warning: save_dir is not specified")
        dir_name = "multi_files_default"
    if not args.sort_model:
        dir_name += "_unsorted"
    if not args.keep_ties:
        dir_name += "_notie"
    save_dir = os.path.join(args.save_base_dir, dir_name)
    os.makedirs(save_dir, exist_ok=args.overwrite)
    print("Saving results to ", save_dir)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # config output logging
    sys.stdout = utils.DualLogger(os.path.join(save_dir, "output.log"))

    # Load data
    conv_a_reward, conv_b_reward, model_a_names, model_b_names, gt_scores = load_rm_rewards_v2(args.rm_dict_file_path)

    rm_scores = [get_rm_score(ra, rb, use_bt_modelling = args.bt) for ra, rb in zip(conv_a_reward, conv_b_reward)]

    # process data
    grouper = utils.ResultGrouperV2(
        model_a_arr = model_a_names,
        model_b_arr = model_b_names,
        rm_scores = rm_scores,
        gt_scores = gt_scores,
        keep_ties = args.keep_ties
    )

    rm_dict, gt_dict = grouper.rm_dict, grouper.gt_dict
    with open(os.path.join(save_dir, "rm_dict.pkl"), "wb") as f:
        pickle.dump(rm_dict, f)
    with open(os.path.join(save_dir, "gt_dict.pkl"), "wb") as f:
        pickle.dump(gt_dict, f)

    for eval_type in ["human", "rm", "cv4"]:
        # initialize evaluators
        arg_list = []
        for model_pair in rm_dict.keys():
            fixed_pair_rm_scores = np.asarray(rm_dict[model_pair])
            fixed_pair_gt_scores = np.asarray(gt_dict[model_pair])
            # Ensure the number of samples is larger than the threshold. 
            # Although rm, human can run below threshold, it will cause problems in future stats.
            if len(fixed_pair_gt_scores) < args.threshold:
                evaluator = None
            elif eval_type == "rm":
                evaluator = RmOnlyEvaluator(fixed_pair_rm_scores)
            elif eval_type == "human":
                evaluator = HumanOnlyEvaluator(fixed_pair_gt_scores)
            elif eval_type == "cv":
                evaluator = ControlVariatesEvaluator(
                    gt_scores = fixed_pair_gt_scores, 
                    rm_scores = fixed_pair_rm_scores, 
                    num_corr_samples = args.num_corr_samples)
            elif eval_type == "cv2":
                evaluator = ControlVariatesEvaluatorV2(
                    gt_scores = fixed_pair_gt_scores, 
                    rm_scores = fixed_pair_rm_scores, 
                    num_corr_samples = args.num_corr_samples)   
            elif eval_type == "cv3":
                evaluator = ControlVariatesEvaluatorV3(
                    gt_scores = fixed_pair_gt_scores, 
                    rm_scores = fixed_pair_rm_scores, 
                    num_corr_samples = args.num_corr_samples) 
            elif eval_type == "cv4":
                evaluator = ControlVariatesEvaluatorV4(
                    gt_scores = fixed_pair_gt_scores, 
                    rm_scores = fixed_pair_rm_scores)            
            else:
                raise ValueError("Invalid eval_type")
            if evaluator is not None:
                arg_list.append((
                    model_pair, 
                    eval_type, 
                    evaluator, 
                    len(fixed_pair_gt_scores), 
                    np.mean(fixed_pair_gt_scores)))
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            result_list = pool.starmap(evaluate_pair, arg_list)
            

        '''
        Result structure:
        {
            <model_pair>: {
                'num_samples': int,
                'pred_type': str (rm, human, cv),
                'prediction': {
                    <num_gt_samples>: List[float],
                    ...
                }
                'gt_winrate': float
            }
        }
        '''
        result_dict = {}
        cnt = 0
        for model_pair, fix_pair_result in result_list:
            if fix_pair_result is None:
                continue
            result_dict[model_pair] = fix_pair_result
            cnt += 1

        print(f"{cnt} pairs have more than {args.num_corr_samples} samples.")

        # save results
        if "cv" in eval_type and eval_type != "cv4":
            save_file_name = f"result_dict_{eval_type}_corr_{args.num_corr_samples}.pkl"
        else:
            save_file_name = f"result_dict_{eval_type}.pkl"
        save_path = os.path.join(save_dir, save_file_name)
        with open(save_path, "wb") as f:
            pickle.dump(result_dict, f)
        print(f"Save result_dict to {save_path}")

    
