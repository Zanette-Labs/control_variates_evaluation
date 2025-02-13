from typing import List, Dict
import numpy as np

class MtbenchDataConversion:
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

        # Process conversation
        if datum["turn"] == 1:
            conv_a = datum["conversation_a"]
            trunc_conv_a = conv_a[:2]
            datum["conversation_a"] = trunc_conv_a
            conv_b = datum["conversation_b"]
            trunc_conv_b = conv_b[:2]
            datum["conversation_b"] = trunc_conv_b
        return datum

class MtbenchSortedDataConversionV2:
    def __init__(self):
        pass
    def __call__(self, datum: Dict):
        # pass
        winner = datum["winner"]
        if "tie" in winner:
            score = 0.5
        elif winner == "model_a":
            score = 1.
        elif winner == "model_b":
            score = 0.
        else:
            raise NotImplementedError
        
        # sort by model name
        model_a = datum["model_a"]
        model_b = datum["model_b"]

        if model_a > model_b: # flip
            datum["model_a"], datum["model_b"] = model_b, model_a
            conv_a = datum["conversation_a"]
            conv_b = datum["conversation_b"]
            datum["conversation_a"], datum["conversation_b"] = conv_b, conv_a
            score = 1. - score

        datum["score"] = score

        # Process conversation
        if datum["turn"] == 1:
            conv_a = datum["conversation_a"]
            trunc_conv_a = conv_a[:2]
            datum["conversation_a"] = trunc_conv_a
            conv_b = datum["conversation_b"]
            trunc_conv_b = conv_b[:2]
            datum["conversation_b"] = trunc_conv_b
        return datum
 