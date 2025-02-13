import torch
import numpy as np
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from collections import defaultdict
from rlaihf.utils import ApiDataConversion
import argparse
import os, json
import sys
from typing import List

def load_gt_data(dataset_path: str, split: str):
    '''
    Load ground truth scores and model names
    '''
    gt_data = datasets.load_dataset(dataset_path, split=split)
    return gt_data
  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, default="results/chatbot-arena_gpt4")
    parser.add_argument('--raw_data_name', type=str, default="annotated_data_all.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Load data
    gt_dataset = load_gt_data("lmsys/chatbot_arena_conversations", split = "train")
    converted_data_path = os.path.join(args.raw_data_dir, "converted_dict.pkl")
    if os.path.exists(converted_data_path):
        print("Load converted data from ", converted_data_path)
        with open(converted_data_path, "rb") as f:
            converted_data = pickle.load(f)
    else:
        print("Convert data")
        converter = ApiDataConversion(gt_dataset, args.raw_data_dir, args.raw_data_name)
        converted_data = converter.convert_and_save()