import pickle 
import argparse
import os
import random
import numpy as np
import torch

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch (CPU and GPU).

    Args:
        seed (int): Random seed value.
    """
    # Python built-in random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = None
        try:
            data = pickle.load(f)
        except EOFError as e:
            data = None
            print(f"pickle file loading error: {e}")
            
def write_pickle(file_path, data):
    with open(file_path, 'wb') as f:
            pkl.dump(data, f)

def append_to_pickle(file_path, new_item):
    # If file doesn't exist, start a new list
    if not os.path.exists(file_path):
        data = []
    else:
        loaded = load_pickle(file_path)
        data = loaded if loaded is not None else []

    # Append the new item
    data.append(new_item)

    # Write back the full list
    write_pickle(file_path, data)

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--data_path", type=str, default='../processed_data')
    parser.add_argument("--seq_length", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--data_limit", action="store_true", help="slice data for testing")
    parser.add_argument("--hidden_size", type=int, default=64)


    parser.add_argument(
        "--positional_features",
        nargs="+",
        default=[],
        help="List of positional feature names"
    )

    parser.add_argument(
        "--time_series_prefixes",
        nargs="+",
        default=["AE_INDEX", "flow_speed", "SYM_H", "Pressure"],
        help="List of time series variable prefixes"
    )
    
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    
    if args.channel == 3:
        args.positional_features = ["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"]
    return parser, args
