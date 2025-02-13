# From RewardBench codebase

import torch

def torch_dtype_mapping(dtype_str):
    """
    Helper function for argparse to map string to torch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise argparse.ArgumentTypeError(f"Invalid torch dtype: {dtype_str}")
    return dtype_map[dtype_str]