import os
import numpy as np
import torch
from common.utils import load_pickle
from torch.utils.data import TensorDataset, DataLoader


LOADER_SAVE_DIR = "dataset/loader"

def load_data(args):
    # load pickle files
    extract_dir = args.data_path
    X_train = load_pickle(os.path.join(extract_dir, 'X_train_norm.pkl'))
    y_train = load_pickle(os.path.join(extract_dir, 'y_train.pkl'))

    X_val = load_pickle(os.path.join(extract_dir, 'X_val_norm.pkl'))
    y_val = load_pickle(os.path.join(extract_dir, 'y_val.pkl'))

    #Out of sample storm data
    X_test_storm = load_pickle(os.path.join(extract_dir, 'X_test_storm_norm.pkl'))
    y_test_storm = load_pickle(os.path.join(extract_dir, 'y_test_storm.pkl'))

    
    # extract relevant features
    positional_features = args.positional_features
    time_series_prefixes = args.time_series_prefixes
    
    columns_to_keep = []

    columns_to_keep.extend(positional_features)

    for prefix in time_series_prefixes:
        columns_to_keep.append(f"{prefix}_t_0")

    X_train_selected = X_train[columns_to_keep]
    X_val_selected = X_val[columns_to_keep]
    X_test_storm_selected = X_test_storm[columns_to_keep]
    
    data = [('train', X_train_selected, y_train), ('validation', X_val_selected, y_val), ('test', X_test_storm_selected, y_test_storm)]
    
    data_loaders = [get_dataloader(args, name, X, y) for name, X, y in data]
    return data_loaders
    
    
def get_dataloader(args, name, X, y):
    print(f"Generating {name} data loader")
    SEQ_LENGTH = args.seq_length  # sequence window size
    BATCH_SIZE = args.batch_size
    
    if args.data_limit:
        X = X.iloc[:20000]
        y = y.iloc[:20000]

    # 1. Prepare sequences
    def create_sequences(x_df, y_df, seq_length):
        xs = []
        ys = []
        for i in range(len(x_df) - seq_length):
            if (i % 10000 == 0):
                print(f"{i}/{len(x_df) - seq_length}")
            x = x_df.iloc[i:i+seq_length, :]  # all features except target flux column
            y = y_df.iloc[i+seq_length]     # flux value at next timestep
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


    X, y = create_sequences(X, y, SEQ_LENGTH)
    
    


    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Dataset and DataLoader

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader



    
