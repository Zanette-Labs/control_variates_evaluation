from typing import List, Dict

class InverseDataConversion:
    '''
    Convert standard dialog format back to "prompt", "response_1", "response_2" format
    every value of key in <dialog_keys> is a standard format dialog. The prompts are the same for every dialog
    Assume every dialog is only single turn.
    '''
    def __init__(self, dialog_keys: List[str]):
        self.dialog_keys = dialog_keys
    def __call__(self, datum: Dict):
        for i, key in enumerate(self.dialog_keys):
            dialog = datum[key]
            if i == 1:
                prompt = dialog[0]["content"]
                datum["prompt"] = prompt
            response = dialog[1]["content"]
            datum[f"response_{i+1}"] = response
        return datum

class ChatBotArenaTrainDataConversion:
    # Convert winner to score
    def __init__(self):
        pass
    def __call__(self, datum: Dict):
        winner = datum["winner"]
        if "tie" in winner:
            score = 0.5
        elif winner == "model_a":
            score = 1.
        elif winner == "model_b":
            score = 0.
        else:
            raise NotImplementedError
        datum["score"] = score
        return datum
        