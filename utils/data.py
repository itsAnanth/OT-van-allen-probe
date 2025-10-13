import pickle as pkl
import numpy as np
import torch

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def load_dataset(file_path, split, channel):
    x = None
    y = None
    
    columns_to_keep = []
    feature_map = {
        '54': {
            'positional_Features': ["ED_R_OP77Q_intxt","ED_MLAT_OP77Q_intxt","ED_MLT_OP77Q_intxt_sin","ED_MLT_OP77Q_intxt_cos"],
            'time_series_features': ["AE_INDEX","flow_speed","SYM_H","Pressure"]
        }
    }


    if split == 'train':
        xfile = 'X_train_norm.pkl'
        yfile = 'y_train.pkl'
    elif split == 'val':
        xfile = 'X_val_norm.pkl'
        yfile = 'y_val.pkl'
    elif split == 'test':
        xfile = 'X_test_norm.pkl'
        yfile = 'y_test.pkl'

    x = load_pickle(f"{file_path}/{xfile}")
    y = load_pickle(f"{file_path}/{yfile}")

    positional_features = feature_map[channel]['positional_features']
    time_series_prefixes = feature_map[channel]['time_series_prefixes']

    columns_to_keep.extend(positional_features)

    for prefix in time_series_prefixes:
        columns_to_keep.append(f"{prefix}_t_0")

    x_selected = x[columns_to_keep]

    return x_selected, y

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


def create_loader(file_path, split, channel, seq_length, batch_size):
    x_selected, y_selected = load_dataset(file_path, split, channel)
    X, y = create_sequences(x_selected.iloc[::1, :], y_selected.iloc[::1], seq_length)

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Dataset and DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader



