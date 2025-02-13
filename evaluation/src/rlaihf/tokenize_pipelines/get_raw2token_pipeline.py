import transformers
from typing import Callable, Optional, List

def get_raw2token_pipeline(tokenizer: transformers.AutoTokenizer, raw2std: Optional[Callable] = None) -> Callable:
    '''
    Return a pipeline of Raw data -> Std data -> formatted data (applied chat template) -> tokenized data.
    Std data format:
        Dict {
        <key_1>: <dialogue_1>,
        <key_2>: <dialogue_2>,
        ...
        <key_n>: <dialogue_n>
        ... (other key-value pairs)
        }
    Tokenized data format:
        Dict {
        <key_1>: <tokenized_dialogue_1>,
        <key_2>: <tokenized_dialogue_2>,
        ...
        <key_n>: <tokenized_dialogue_n>
        ... (other key-value pairs)
        }

    dialogue std format:
        [
        {'role': 'user', 'content': 'xxx'},
        {'role': 'assistant', 'content': 'xxx'},
        ...
        ]
    
    Parameters:
    raw2std: A method that converts raw dialogue to std dialogue format
    
    '''
    if raw2std is None:
        def tokenize_from_std(std_datum, dialogue_keys: List[str] = ['chosen', 'rejected']):
            '''
            The datum is already in std format. Tokenize items with key in `dialogue_keys`
            '''
            for k in dialogue_keys:
                template = tokenizer.apply_chat_template(std_datum[k], tokenize = False)
                tokenized_dialogue = tokenizer(template, return_tensors="pt", padding = False)
                std_datum[k] = tokenized_dialogue
            return std_datum
        return tokenize_from_std
    else:
        def tokenize_from_raw(raw_datum, dialogue_keys: List[str] = ['chosen', 'rejected']):
            for k in dialogue_keys:
                template = tokenizer.apply_chat_template(raw2std(raw_datum[k]), tokenize = False)
                tokenized_dialogue = tokenizer(template, return_tensors="pt", padding = False)
                raw_datum[k] = tokenized_dialogue
            return raw_datum    
        return tokenize_from_raw

# Same functionality as get_raw2token_pipeline, but can be passed as argument using LightningCLI
class TokenizePipeline:
    '''
    Return a pipeline of Raw data -> Std data -> formatted data (applied chat template) -> tokenized data.
    Std data format:
        Dict {
        <key_1>: <dialogue_1>,
        <key_2>: <dialogue_2>,
        ...
        <key_n>: <dialogue_n>
        ... (other key-value pairs)
        }
    Tokenized data format:
        Dict {
        <key_1>: <tokenized_dialogue_1>,
        <key_2>: <tokenized_dialogue_2>,
        ...
        <key_n>: <tokenized_dialogue_n>
        ... (other key-value pairs)
        }

    dialogue std format:
        [
        {'role': 'user', 'content': 'xxx'},
        {'role': 'assistant', 'content': 'xxx'},
        ...
        ]
    
    Parameters:
    raw2std: A method that converts raw dialogue to std dialogue format
    
    '''
    def __init__(self, model_name_or_path: str, raw2std: Optional[Callable] = None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.raw2std = raw2std
    def __call__(self, datum, dialogue_keys: List[str] = ['chosen', 'rejected']):
        for k in dialogue_keys:
            if self.raw2std is not None:
                std_datum = self.raw2std(anthropic_dialog=datum[k])
                template = self.tokenizer.apply_chat_template(std_datum, tokenize = False)
            else:
                template = self.tokenizer.apply_chat_template(datum[k], tokenize = False)
            tokenized_dialogue = self.tokenizer(template, return_tensors="pt", padding = False)
            datum[k] = tokenized_dialogue
        return datum 

class TokenizePipelineV2:
    '''
    Comparing to v1: use data_convert instead of dialog convert to make conversion more general
    Return a pipeline of Raw data -> Std data -> formatted data (applied chat template) -> tokenized data.
    Std data format:
        Dict {
        <key_1>: <dialogue_1>,
        <key_2>: <dialogue_2>,
        ...
        <key_n>: <dialogue_n>
        ... (other key-value pairs)
        }
    Tokenized data format:
        Dict {
        <key_1>: <tokenized_dialogue_1>,
        <key_2>: <tokenized_dialogue_2>,
        ...
        <key_n>: <tokenized_dialogue_n>
        ... (other key-value pairs)
        }

    dialogue std format:
        [
        {'role': 'user', 'content': 'xxx'},
        {'role': 'assistant', 'content': 'xxx'},
        ...
        ]
    
    Parameters:
    raw_data_convert: A method that converts raw data (usually convert dialogue to std format)
    dialog_keys: List of keys with values being dialogs to tokenize
    
    '''
    def __init__(self, model_name_or_path: str, dialog_keys: List[str], raw_data_convert: Optional[Callable] = None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.raw_data_convert = raw_data_convert
        self.dialog_keys = dialog_keys
    def __call__(self, datum):
        if self.raw_data_convert is not None:
            datum = self.raw_data_convert(datum)
        for k in self.dialog_keys:
            template = self.tokenizer.apply_chat_template(datum[k], tokenize = False)
            tokenized_dialogue = self.tokenizer(template, return_tensors="pt", padding = False)
            datum[k] = tokenized_dialogue

        # for k in dialogue_keys:
        #     if self.raw2std is not None:
        #         std_datum = self.raw2std(anthropic_dialog=datum[k])
        #         template = self.tokenizer.apply_chat_template(std_datum, tokenize = False)
        #     else:
        #         template = self.tokenizer.apply_chat_template(datum[k], tokenize = False)
        #     tokenized_dialogue = self.tokenizer(template, return_tensors="pt", padding = False)
        #     datum[k] = tokenized_dialogue
        return datum 

class NoTokenizePiepline:
    '''
    Do not do tokenization. It's actually just a data converter. Implement this class to make it compatible to datamodule. Useful for OpenAI API calls.
    Return a pipeline of Raw data -> Std data.
    Std data format:
        Dict {
        <key_1>: <dialogue_1>,
        <key_2>: <dialogue_2>,
        ...
        <key_n>: <dialogue_n>
        ... (other key-value pairs)
        }

    dialogue std format:
        [
        {'role': 'user', 'content': 'xxx'},
        {'role': 'assistant', 'content': 'xxx'},
        ...
        ]
    
    Parameters:
    raw_data_convert: A method that converts raw data (usually convert dialogue to std format)
    dialog_keys: List of keys with values being dialogs to tokenize
    
    '''
    def __init__(self, raw_data_convert: Optional[Callable] = None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    def __call__(self, datum):
        if self.raw_data_convert is not None:
            datum = self.raw_data_convert(datum)
        return datum 