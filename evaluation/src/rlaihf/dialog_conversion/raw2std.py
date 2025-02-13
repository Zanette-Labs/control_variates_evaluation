from typing import List, Dict

def anthropic2std(anthropic_dialog: str) -> List[Dict[str,str]]:
    '''
    Convert Anthropic/hh-rlhf data into standard raw data format. 
    Raw format: "Human: xxx Assistant: xxx ..."
    '''
    human_splitted_dialogs = anthropic_dialog.split("\n\nHuman:")
    # print(human_splitted_dialogs)
    dict_dialog = []
    for idx in range(1, len(human_splitted_dialogs)): # First string is empty
        human_assistant_dialog = human_splitted_dialogs[idx]
        split_human_assistant_dialogs = human_assistant_dialog.split("\n\nAssistant:")
        if len(split_human_assistant_dialogs) < 2:
            continue
        dict_dialog.append({
            'role': 'user',
            'content': split_human_assistant_dialogs[0]
        })
        # Not the end but empy
        if split_human_assistant_dialogs[1] == ' ' and idx < len(human_splitted_dialogs)-1: 
            buggy_text = human_splitted_dialogs[idx+1]
            assistant_text = buggy_text.split("Assistant:")[0]
        else:
            assistant_text = split_human_assistant_dialogs[1]
        dict_dialog.append({
            'role': 'assistant',
            'content': assistant_text
        })
    return dict_dialog

class Anthropic2Std:
    def __init__(self):
        pass
    def __call__(self, anthropic_dialog: str) -> List[Dict[str,str]]:
        human_splitted_dialogs = anthropic_dialog.split("\n\nHuman:")
        dict_dialog = []
        for idx in range(1, len(human_splitted_dialogs)): # First string is empty
            human_assistant_dialog = human_splitted_dialogs[idx]
            split_human_assistant_dialogs = human_assistant_dialog.split("\n\nAssistant:")
            if len(split_human_assistant_dialogs) < 2:
                continue
            dict_dialog.append({
                'role': 'user',
                'content': split_human_assistant_dialogs[0]
            })
            # Not the end but empy
            if split_human_assistant_dialogs[1] == ' ' and idx < len(human_splitted_dialogs)-1: 
                buggy_text = human_splitted_dialogs[idx+1]
                assistant_text = buggy_text.split("Assistant:")[0]
            else:
                assistant_text = split_human_assistant_dialogs[1]
            dict_dialog.append({
                'role': 'assistant',
                'content': assistant_text
            })
        return dict_dialog      

class PromptResponse2Std:
    '''
    Convert dialogue with a seperate prompt and response
    '''
    def __init__(self):
        pass

    def __call__(self, prompt: str, response: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]