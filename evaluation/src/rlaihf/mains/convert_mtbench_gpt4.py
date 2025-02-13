# Based on https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/compute_agreement.py
from rlaihf.data_conversion import MtbenchSortedDataConversion

import datasets
import numpy as np
import pandas as pd
import pickle
import os

def get_judge_name(judge, discern_human = True):
    if isinstance(judge, list) and judge[0] == "gpt-4" and judge[1].startswith("pair"):
        return "gpt4-pair"
    if judge.startswith("expert"):
        return "human"
    if judge.startswith("author") and discern_human:
        return "author"
    if judge.startswith("author") and not discern_human:
        return "human"    
    return judge

def revert(vote):
    if vote == "model_a":
        return "model_b"
    elif vote == "model_b":
        return "model_a"
    return vote

def get_mt_bench_votes_data(raw_votes, discern_human = True):
    '''
    Return: List of length 2, elements are data in turn 1 and turn 2
    Each turn data is a dict, with keys being question id and the models evaluated. 
    The value is a dict, with keys being the annotator (author, human, or gpt4_pair), and values being a list of annoations.
    '''
    data = [{}, {}]

    for judge_votes in raw_votes:
        for vote in judge_votes:
            turn = vote["turn"] - 1
            if vote["model_a"] < vote["model_b"]:
                key = (vote["question_id"], vote["model_a"], vote["model_b"])
                winner = vote["winner"]
            else:
                key = (vote["question_id"], vote["model_b"], vote["model_a"])
                winner = revert(vote["winner"])
            judge = get_judge_name(vote["judge"], discern_human=discern_human)
            if key not in data[turn]:
                data[turn][key] = {}
            if judge not in data[turn][key]:
                data[turn][key][judge] = []
            data[turn][key][judge].append(winner)

    return data


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

    # Convert the grouped DataFrame back to datasets.Dataset
    new_pd_dataset = grouped_df.reset_index()
    return new_pd_dataset

def merge_data(data_human, data_gpt):
    '''
    Load, tokenize, save to disk.
    '''
    data_human = data_human.map(MtbenchSortedDataConversion(), batched=False)
    data_gpt = data_gpt.map(MtbenchSortedDataConversion(), batched=False)

    data_human_avg_df = process_mtbench_dataset(data_human)
    data_gpt_df = process_mtbench_dataset(data_gpt)
    data_human_avg_df = data_human_avg_df.rename(columns={"score": "gt_score"})
    data_gpt_df = data_gpt_df.rename(columns={"score": "api_score"})
    merge_cols = ['question_id', 'model_a', 'model_b', 'turn']
    merged_df = pd.merge(
        data_human_avg_df,
        data_gpt_df[merge_cols + ['api_score']],
        on=merge_cols,
        how='left'
    )
    return merged_df

if __name__ == "__main__":

    data_human = datasets.load_dataset("lmsys/mt_bench_human_judgments", split="human")
    data_gpt = datasets.load_dataset("lmsys/mt_bench_human_judgments", split="gpt4_pair")
    '''
        [{[(question_id, model_a, model_b)]["gpt4_pair"/"human"]}, 
        {[(question_id, model_a, model_b)]["gpt4_pair"/"human"]}]
        model names are already sorted
    '''
    data_dict_two_turns = get_mt_bench_votes_data([data_human, data_gpt], discern_human = False)
    merged_df = merge_data(data_human, data_gpt)
    merged_dict = merged_df.to_dict(orient='list')
    n = len(merged_dict['question_id'])
    merged_dict['id'] = np.arange(n)
    for key in ["model_a", "model_b", "api_score", "gt_score"]:
        merged_dict[key] = np.asarray(merged_dict[key])
    os.makedirs("results/mt-bench_gpt4", exist_ok=True)
    with open("results/mt-bench_gpt4/converted_dict.pkl", "wb") as f:
        pickle.dump(merged_dict, f)