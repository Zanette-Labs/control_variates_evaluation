from datasets import load_dataset, Dataset
from typing import Dict

class DatasetLoader:
    '''
    predefine ways to load specific datasets
    '''
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.loaded_dataset = self.load_dataset()

    def load_dataset(self):
        if self.dataset_name == "chatbot_arena":
            return load_dataset("lmsys/chatbot_arena_conversations", split="train")
        if self.dataset_name == "mt_bench":
            dataset = load_dataset("lmsys/mt_bench_human_judgments", split="human")
            dataset = dataset.map(MtbenchDataConversion(), batched=False)
            return dataset
        raise NotImplementedError

class MtbenchDataConversion:
    # Process single turn conversation
    # Sort model pair
    # Convert winner to score
    def __init__(self):
        pass
    def __call__(self, datum: Dict):
        # Process conversation
        if datum["turn"] == 1:
            conv_a = datum["conversation_a"]
            trunc_conv_a = conv_a[:2]
            datum["conversation_a"] = trunc_conv_a
            conv_b = datum["conversation_b"]
            trunc_conv_b = conv_b[:2]
            datum["conversation_b"] = trunc_conv_b
        return datum


def transform_dataset(original_dataset, keep_tie: bool = False):
    chosen = []
    rejected = []
    model_chosen = []
    model_rejected = []
    is_tie = []
    for example in original_dataset:
        if example['winner'] == 'model_a':
            chosen.append(example['conversation_a'])
            rejected.append(example['conversation_b'])
            model_chosen.append(example['model_a'])
            model_rejected.append(example['model_b'])
            is_tie.append(False)
        elif example['winner'] == 'model_b':
            chosen.append(example['conversation_b'])
            rejected.append(example['conversation_a'])
            model_chosen.append(example['model_b'])
            model_rejected.append(example['model_a'])
            is_tie.append(False)
        elif keep_tie:
            # Use opposite labels for the tie case
            chosen.append(example['conversation_a'])
            rejected.append(example['conversation_b'])
            model_chosen.append(example['model_a'])
            model_rejected.append(example['model_b'])
            is_tie.append(True)

            chosen.append(example['conversation_b'])
            rejected.append(example['conversation_a'])
            model_chosen.append(example['model_b'])
            model_rejected.append(example['model_a'])
            is_tie.append(True)

            # ties += 2
        # Skip the entry if the winner is 'tie'

    # Create and return a new Dataset with 'chosen' and 'rejected' columns
    transformed_dataset = Dataset.from_dict({
        'chosen': chosen, 
        'rejected': rejected,
        'model_chosen': model_chosen,
        'model_rejected': model_rejected,
        'is_tie': is_tie})
    return transformed_dataset

def is_train_example(example, holdout_model):
    # Keep examples where neither model matches the holdout_model
    return example['model_chosen'] != holdout_model and example['model_rejected'] != holdout_model

def is_test_example(example, holdout_model):
    # Keep examples where at least one model matches the holdout_model
    # print(example['model_chosen'], example['model_rejected'])
    # print(f"Holdout: {holdout_model}")
    # print(example['model_chosen'] == holdout_model)
    return example['model_chosen'] == holdout_model or example['model_rejected'] == holdout_model


def prepare_dataset(dataset_name: str, holdout_model_id: int, keep_tie: bool = False):
    '''
    Load and rename columns
    '''
    # dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    # dataset = dataset.select(range(100))
    print(f"Loading dataset {dataset_name}")
    loaded_dataset = DatasetLoader(dataset_name).loaded_dataset
    # print(f"Original dataset: {len(loaded_dataset)}")
    new_dataset = transform_dataset(loaded_dataset, keep_tie)
    # print(f"New dataset: {len(new_dataset)}")

    # Split the dataset
    model_a = new_dataset['model_chosen']
    model_b = new_dataset['model_rejected']
    model_set = set(model_a + model_b)
    model_list = sorted(list(model_set))
    print(f"Number of models: {len(model_list)}")
    holdout_model = model_list[holdout_model_id]
    print(f"Holdout model: {holdout_model}")

    # Example usage:
    # Assuming 'dataset' is your original Dataset object
    # holdout_model = "example_model_name"

    # Filter the train and test datasets
    train_dataset = new_dataset.filter(lambda example: is_train_example(example, holdout_model))
    test_dataset = new_dataset.filter(lambda example: is_test_example(example, holdout_model))
    # print(f"Train dataset: {len(train_dataset)}")
    # print(f"Test dataset: {len(test_dataset)}")
    # print(f"Ties in test dataset: {sum(test_dataset['is_tie'])}")
    # train_dataset.to_json("train_dataset.json")
    # test_dataset.to_json("test_dataset.json")
    return train_dataset, test_dataset
