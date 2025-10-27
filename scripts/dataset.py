import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from common.utils import load_pickle
from torch.utils.data import TensorDataset, DataLoader
from common.config import Config



def load_data(args: Config):
        
    # load pickle files
    extract_dir = f"{args.data_dir}/{args.channel}"
    X_train = load_pickle(os.path.join(extract_dir, 'X_train_norm.pkl'))
    y_train = load_pickle(os.path.join(extract_dir, 'y_train.pkl'))

    X_val = load_pickle(os.path.join(extract_dir, 'X_val_norm.pkl'))
    y_val = load_pickle(os.path.join(extract_dir, 'y_val.pkl'))

    #Out of sample storm data
    X_test_storm = load_pickle(os.path.join(extract_dir, 'X_test_storm_norm.pkl'))
    y_test_storm = load_pickle(os.path.join(extract_dir, 'y_test_storm.pkl'))
    

    
    # extract relevant features
    positional_features = args.channel_data['positional_features']
    time_series_prefixes = args.channel_data['time_series_features']
    
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
    
    



class LazySequenceDataset(Dataset):
    def __init__(self, X_df, y_df, seq_length, convert_to_numpy=True):
        self.seq_length = seq_length
        self.length = len(X_df) - seq_length
        
        if convert_to_numpy:
            print("Converting to numpy arrays (one-time operation)...")
            self.X_data = X_df.values.astype(np.float32)
            
            # Handle different y shapes
            if isinstance(y_df, pd.Series):
                self.y_data = y_df.values.astype(np.float32)
            else:
                self.y_data = y_df.astype(np.float32)
            
            self.use_numpy = True
        else:
            self.X_data = X_df
            self.y_data = y_df
            self.use_numpy = False
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.use_numpy:
            x = self.X_data[idx:idx+self.seq_length]
            y = self.y_data[idx+self.seq_length]
        else:
            x = self.X_data.iloc[idx:idx+self.seq_length].values
            y = self.y_data.iloc[idx+self.seq_length]
        
        # FIX: Return scalar y, not wrapped in list
        # This ensures y has shape [batch_size] after batching
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
    
def get_dataloader(args: Config, name, X, y, create_sequence=True):
    print(f"Generating {name} data loader")
    SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch_size

    if args.data_limit:
        X = X.iloc[:20000]
        y = y.iloc[:20000]

    # Use lazy loading - no sequence creation upfront
    dataset = LazySequenceDataset(X, y, SEQ_LENGTH)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2
    )
    return loader



    
