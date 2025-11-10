import argparse
import py7zr
import os
import pickle as pkl
import pandas as pd
import numpy as np
import zipfile

channels = {
    3: 'ch3_data',
    11: 'ch11_data',
    14: 'ch14_data',
    16: 'ch16_data',
    'meta': 'ORIENT-M_model.zip'
}
ARCHIVE_DIR = 'dataset/raw'
EXTRACT_DIR = 'dataset/extracted'
output_dir = 'dataset/preprocessed'

def extract(args):

    if args.channel == -1:
        
        for key, value in channels.items():
            print(f"Extracting {value} to {EXTRACT_DIR}")
            
            if key == 'meta':
                with zipfile.ZipFile(f"{ARCHIVE_DIR}/{value}", 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_DIR)
                continue
            with py7zr.SevenZipFile(f"{ARCHIVE_DIR}/{value}.7z", mode='r') as archive:
                archive.extractall(path=f"{EXTRACT_DIR}")

                

def label_spacecraft(df, time_col='unix_time'):
    t = df[time_col].astype(float).values
    diffs = np.diff(t)
    neg_jumps = np.where(diffs < 0)[0]

    df = df.copy()
    if len(neg_jumps) > 0:
        split_idx = neg_jumps[0] + 1
        df.loc[:split_idx-1, 'spacecraft'] = 'A'
        df.loc[split_idx:, 'spacecraft'] = 'B'
        print(f"A→B transition at index {split_idx}")
    else:
        df['spacecraft'] = 'A'
        print("No A→B transition found (all labeled 'A')")
    return df
                
def preprocess(args):
    for key, value in channels.items():
        
        if key == 'meta':
            continue
            
        print(f"Preprocessing {value}.pkl")
        df = pd.read_pickle(f"{EXTRACT_DIR}/{value}.pkl")


        if key == 16:
            df = df.sort_index()
            print("Sorted index for ch16.")

        # Check if the index is sorted
        is_index_sorted = df.index.is_monotonic_increasing
        print(f"Is index increasing? {is_index_sorted}")

        df['datetime'] = pd.to_datetime(df['unix_time'], unit='s') #to datetime conversion

        df = label_spacecraft(df, time_col='unix_time')

        df.set_index('datetime', inplace=True) #new index
        df.sort_index(inplace=True) #sorting for splitting later
        df['target'] = np.log10(df['eflux'] + 1) #log transformation to eflux column along with handling 0 by adding 1
        y = df['target'] # setting target variable
        positional_cols = [
            'ED_R_OP77Q_intxt',
            'ED_MLAT_OP77Q_intxt',
            'ED_MLT_OP77Q_intxt_sin',
            'ED_MLT_OP77Q_intxt_cos'
        ]
        driver_cols = [col for col in df.columns if '_t_' in col] #all the features with the time-history suffix '_t_'
        
        feature_cols = positional_cols + driver_cols
        X = df[feature_cols] #input features
        
        full_data = pd.concat([X, y], axis=1)
        full_data.dropna(inplace=True) #drop na columns


        y = full_data['target']
        X = full_data.drop(columns=['target'])
        
        train_end_date = '2016-12-31'
        val_start_date = '2017-01-01'
        val_end_date = '2017-02-24'
        storm_start_date = '2017-02-25'
        storm_end_date = '2017-03-25'
        
        X_train = X.loc[:train_end_date]
        y_train = y.loc[:train_end_date]
        
        X_val = X.loc[val_start_date:val_end_date]
        y_val = y.loc[val_start_date:val_end_date]
        
        X_test_storm = X.loc[storm_start_date:storm_end_date]
        y_test_storm = y.loc[storm_start_date:storm_end_date]
        
        print(f"Training set size:   {len(X_train)} samples")
        print(f"Validation set size: {len(X_val)} samples")
        print(f"Storm test set size: {len(X_test_storm)} samples")
        
        
        avg_file = f'{EXTRACT_DIR}/ORIENT-M_model/mageis_{value.split("_")[0]}_input_avg.npy'
        std_file = f'{EXTRACT_DIR}/ORIENT-M_model/mageis_{value.split("_")[0]}_input_std.npy'

        # pre-computed mean and standard deviation
        avg = np.load(avg_file)
        std = np.load(std_file)

        # Z-score normalization to all feature sets
        X_train_norm = (X_train - avg) / std
        X_val_norm = (X_val - avg) / std
        X_test_storm_norm = (X_test_storm - avg) / std
        
        
        os.makedirs(f"{output_dir}/{key}", exist_ok=True)
        
        X_train_norm.to_pickle(os.path.join(f"{output_dir}/{key}", 'X_train_norm.pkl'))
        X_val_norm.to_pickle(os.path.join(f"{output_dir}/{key}", 'X_val_norm.pkl'))
        X_test_storm_norm.to_pickle(os.path.join(f"{output_dir}/{key}", 'X_test_storm_norm.pkl'))

        # Save the corresponding target sets
        y_train.to_pickle(os.path.join(f"{output_dir}/{key}", 'y_train.pkl'))
        y_val.to_pickle(os.path.join(f"{output_dir}/{key}", 'y_val.pkl'))
        y_test_storm.to_pickle(os.path.join(f"{output_dir}/{key}", 'y_test_storm.pkl'))
        

def parse_args():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--channel", type=int, default=-1)

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    
    assert args.channel in [-1, *channels.keys()], "Invalid channel number"
    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(EXTRACT_DIR):
        print("extract dir already exists, skipping extraction")
    else:
        extract(args)
    preprocess(args)