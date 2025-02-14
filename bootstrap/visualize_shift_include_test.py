import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import os
import argparse
from copy import deepcopy
import json

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_err_single_run(pred_scores, gt_winrate):
    '''
    Compute the mse_err for a list of predicted scores and the ground truth winrate
    '''
    pred_scores = np.asarray(pred_scores)
    # num_nan = np.sum(np.isnan(pred_scores))
    # if num_nan > 500:
    # print(f"Warning: number of nan{num_nan}")
    return np.nanmean((pred_scores - gt_winrate) ** 2)

def compute_err_single_pair(prediction_dict, gt_winrate):
    '''
    For a single model pair, compute the mse_err
    '''
    gt_winrate = prediction_dict['gt_winrate']
    err_dict = {}
    for k, v in prediction_dict['prediction'].items():
        err_dict[k] = compute_err_single_run(v, gt_winrate)
    return err_dict

def compute_avg_err(result_dict, threshold):
    '''
    Compute the average mse_err for all model pairs
    '''
    err_dict_list = []
    for single_pair_result_dict in result_dict.values():
        if single_pair_result_dict["num_samples"] < threshold:
            continue
        gt_winrate = single_pair_result_dict['gt_winrate']
        err_dict = compute_err_single_pair(single_pair_result_dict, gt_winrate)
        err_dict_list.append(err_dict)

    num_gt_sample_list = list(err_dict_list[0].keys())
    avg_err_list = []
    for num_gt_sample in num_gt_sample_list:
        err_list = [err_dict[num_gt_sample] for err_dict in err_dict_list]
        avg_err = np.mean(err_list)
        avg_err_list.append(avg_err)

    return num_gt_sample_list, avg_err_list 

def compute_rm_error(rm_dict, gt_dict, threshold):
    '''
    Compute the average mse error for all model pairs, using standard synthetic evaluation
    '''
    err_list = []
    for model_pair in rm_dict.keys():
        rm_scores = rm_dict[model_pair]
        gt_scores = gt_dict[model_pair]
        if len(rm_scores) < threshold:
            continue
        square_err = (np.mean(rm_scores) - np.mean(gt_scores)) ** 2
        err_list.append(square_err)

    return np.mean(err_list)

def label_convert(eval_type):
    map_dict = {
        'human': "Human",
        'rm': "Automatic",
        'cv4': 'Control Variates'
    }
    return map_dict[eval_type]

def visualize_err(err_dict, synthetic_square_err, save_path: str, fontsize=18):
    color_dict = {
        'human': 'b',
        'rm': 'g',
        'cv4': 'r'
    }
    fig, ax = plt.subplots(figsize=(8, 6)) 
    for eval_type in ['human', 'cv4']:
        err_data = err_dict[eval_type]
        num_gt_sample_list, avg_err_list = err_data

        num_gt_sample_list = np.asarray(num_gt_sample_list)
        avg_err_list = np.asarray(avg_err_list)
        
        # Plot less data
        idx = np.logical_and(num_gt_sample_list <= 1000, 200 <= num_gt_sample_list)
        num_gt_sample_list = num_gt_sample_list[idx]
        avg_err_list = avg_err_list[idx]

        ax.plot(num_gt_sample_list, avg_err_list, color = color_dict[eval_type], label=label_convert(eval_type))

    ax.plot(num_gt_sample_list, np.ones_like(num_gt_sample_list) * synthetic_square_err, color = "black", linewidth = 2, linestyle='--', label="Synthetic")

    ax.set_yscale('log')
    ax.set_yticks([1e-4, 1e-3, 1e-2])
    ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation())
    ax.set_xlabel("Number of Human Annotations", fontsize=fontsize)
    ax.set_ylabel("Averaged MSE", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax.legend(fontsize=fontsize)

    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize)
    plt.savefig(save_path)
    plt.close(fig)

# Copy this function to final version
def visualize_var(err_dict, save_path: str, shift, fontsize=18):
    color_dict = {
        'human': 'b',
        'rm': 'g',
        'cv4': 'r'
    }
    fig, ax = plt.subplots(figsize=(8,6)) 
    for eval_type in ['human', 'cv4']:
        err_data = err_dict[eval_type]
        num_gt_sample_list, avg_err_list = err_data

        num_gt_sample_list = np.asarray(num_gt_sample_list)
        avg_err_list = np.asarray(avg_err_list)
        
        # Plot less data
        idx = np.logical_and(num_gt_sample_list <= 1000, 200 <= num_gt_sample_list)
        num_gt_sample_list = num_gt_sample_list[idx]
        avg_err_list = avg_err_list[idx]

        # Get shifted curve
        if "human" in eval_type:
            human_shift_avg_err_list = deepcopy(avg_err_list)
            human_shift_num_gt_sample_list = deepcopy(num_gt_sample_list) * (1-shift)
            human_idx = human_shift_num_gt_sample_list <= 1000
            human_shift_num_gt_sample_list = human_shift_num_gt_sample_list[human_idx]
            human_shift_avg_err_list = human_shift_avg_err_list[human_idx]
            human_shift_avg_err_list = human_shift_avg_err_list[::40]
            human_shift_num_gt_sample_list = human_shift_num_gt_sample_list[::40]

        ax.plot(num_gt_sample_list, avg_err_list, color = color_dict[eval_type], label=label_convert(eval_type), linewidth=2)

    ax.plot(human_shift_num_gt_sample_list, human_shift_avg_err_list, label="Human (shifted)", color = color_dict['human'], marker="d", linewidth = 1, linestyle='--')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_xlabel("Number of Human Annotations", fontsize=fontsize)
    ax.set_ylabel("Averaged MSE", fontsize=fontsize)
    ax.set_yticks([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax.legend(fontsize=fontsize)

    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize)
    plt.savefig(save_path)
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_root', type=str, default="results/")
    parser.add_argument('-s', '--save_root', type=str, default="results/")
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--model_type', type=str, required=True, help = 'pretrained or finetuned')
    parser.add_argument('--threshold', type=int, default=100)
    parser.add_argument('--ckpt_name', type=str, default="None")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # Get correlation
    with open("saving.json", 'r') as f:
        saving_dict = json.load(f)
    saving = saving_dict[f"{args.dataset}_{args.model}"][args.model_type]
    print(f"saving: {saving}")
    saving = saving / 100 # raw data is in percentage

    # Get run data
    result_base_dir = f"{args.dataset}_{args.model}"
    save_base_dir = f"{args.dataset}_{args.model}"
    if args.model_type == "pretrained":
        result_base_dir += "_pretrained"
    result_file_dir = os.path.join(args.result_root, result_base_dir, args.model_type)
    save_file_dir = os.path.join(args.save_root, save_base_dir, args.model_type)

    # Get rm, gt dict
    with open(os.path.join(result_file_dir, "rm_dict.pkl"), "rb") as f:
        rm_dict = pickle.load(f)
    with open(os.path.join(result_file_dir, "gt_dict.pkl"), "rb") as f:
        gt_dict = pickle.load(f)

    synthetic_square_err = compute_rm_error(rm_dict, gt_dict, args.threshold)

    get_err_dict = True
    if args.ckpt_name != "None":
        err_dict_path = os.path.join(result_file_dir, args.ckpt_name)
        if os.path.exists(err_dict_path):
            print(f"Load err_dict from {err_dict_path}")
            err_dict = load_data(err_dict_path)
            get_err_dict = False
    if get_err_dict:
        err_dict = {}
        for eval_type in ['rm', 'human','cv4']:
            file_path = os.path.join(result_file_dir, f"result_dict_{eval_type}.pkl")
            result_dict = load_data(file_path)
            num_gt_sample_list, avg_err_list = compute_avg_err(result_dict, args.threshold)
            err_dict[eval_type] = (num_gt_sample_list, avg_err_list)
        save_path = os.path.join(result_file_dir, f"err_dict_threshold_{args.threshold}_cv4_rm_human.pkl")
        print(f"Save err_dict to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(err_dict, f)
    
    # Visualize
    os.makedirs(os.path.join(save_file_dir, "images"), exist_ok=True)
    save_err_path = os.path.join(save_file_dir, "images", f"bootstrap_err_logscale.png")
    save_var_path = os.path.join(save_file_dir, "images", f"bootstrap_var.png")
    visualize_err(err_dict, synthetic_square_err, save_err_path)
    visualize_var(err_dict, save_var_path, saving)
    print(f"Save visualization to {save_err_path}, {save_var_path}")
    
        
