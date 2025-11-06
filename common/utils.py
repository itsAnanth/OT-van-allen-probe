import pickle 
import pickle as pkl
import argparse
import os
import random
import numpy as np
import torch
from datetime import datetime, timezone, timedelta
import os




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
    
def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU {tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print("[GPU] CUDA not available.")

def load_pickle(file_path):
    data = None
    
    with open(file_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError as e:
            data = None
            print(f"pickle file loading error: {e}")
            
    return data
            
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
    
    
def save_checkpoint(file_path, file_name, data):
    os.makedirs(file_path, exist_ok=True)
    
    torch.save(data, f"{file_path}/{file_name}")
    
def load_checkpoint(epoch, model, optimizer, config):
    if not os.path.exists(config.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {config.checkpoint_dir} not found")
    
    checkpoint_path = os.path.join(config.checkpoint_dir, f"{config.channel_name}_{epoch}.pth")
    checkpoint_data = torch.load(checkpoint_path, weights_only=True, map_location=config.device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model, optimizer

    
    
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


def get_checkpoints_dir(config):
    IST = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(IST).strftime('%d-%m-%Y_%H-%M-%S')

    
    print(config.checkpoint_dir)
    return f"{config.checkpoint_dir}/{'tune/' if config.tune else ''}{config.channel_name}/{timestamp}"

def autodetect_device():
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'
        
    logger.info(f"Autodetected device type as {device_type}")
    return device_type