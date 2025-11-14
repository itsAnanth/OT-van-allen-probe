import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates
import torch
from tqdm import tqdm
from common.config import Config, CHANNEL_MAP, get_model_class
from common.utils import load_pickle
from matplotlib.markers import MarkerStyle
import argparse

"""
   Usage: python scripts/plot_fluxmap.py --checkpoint-path checkpoints/54kev/07-11-2025_10-27-48/54kev_17.pth
   
   output saved in plots/L_time_fluxmap_LSTM_<channel_name>.png
"""

def restore_l_shell(X_df, raw_path):
    """Restore true L-shell values from raw data by aligning timestamps."""
    print("Restoring L-shell from raw data...")
    raw_df = pd.read_pickle(raw_path)
    raw_df["datetime"] = pd.to_datetime(raw_df["unix_time"], unit="s")
    raw_df.set_index("datetime", inplace=True)
    raw_df.sort_index(inplace=True)

    if "Lm_eq_OP77Q_intxt" not in raw_df.columns:
        raise KeyError("Lm_eq_OP77Q_intxt not found in raw data!")

    merged = pd.merge_asof(
        X_df.sort_index(),
        raw_df[["Lm_eq_OP77Q_intxt"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("10min")
    )
    merged.rename(columns={"Lm_eq_OP77Q_intxt": "L_shell"}, inplace=True)
    return merged


def load_and_prepare_data(config, model, channel, raw_path):
    """
    Load preprocessed test data and run inference.
    Uses the SAME preprocessing as dataset.py:
    1. Load normalized data (already done by preprocess.py)
    2. Filter to relevant features (same as load_data function)
    3. Create sequences in order (not shuffled - needed for time-based plotting)
    """
    # Use same paths as load_data() in dataset.py
    extract_dir = f"{config.data_dir}/{channel}"
    
    print(f"Loading preprocessed test data from {extract_dir}")
    X_test = load_pickle(os.path.join(extract_dir, 'X_test_storm_norm.pkl'))
    y_test = load_pickle(os.path.join(extract_dir, 'y_test_storm.pkl'))
    
    # Extract relevant features (same as load_data function)
    positional_features = config.channel_data['positional_features']
    time_series_prefixes = config.channel_data['time_series_features']
    
    columns_to_keep = []
    columns_to_keep.extend(positional_features)
    for prefix in time_series_prefixes:
        columns_to_keep.append(f"{prefix}_t_0")
    
    X_selected = X_test[columns_to_keep]
    
    # Ensure datetime index
    if not isinstance(X_selected.index, pd.DatetimeIndex):
        X_selected.index = pd.date_range("2017-02-25", periods=len(X_selected), freq="min")
    
    # Restore L-shell from raw data
    X_selected = restore_l_shell(X_selected, raw_path)
    
    # Create sequences IN ORDER (not shuffled like in LazySequenceDataset)
    X_np = X_selected[columns_to_keep].to_numpy(np.float32)
    seq_len = config.seq_length
    
    print(f"Creating sequences (seq_length={seq_len})...")
    seqs = [X_np[i:i+seq_len] for i in range(len(X_np) - seq_len)]
    X_seq = np.stack(seqs)
    
    # Run inference
    print("Running model inference...")
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_seq), 256)):
            batch = torch.from_numpy(X_seq[i:i+256]).to(config.device)
            y_pred = model(batch)
            preds.append(y_pred.cpu().numpy().ravel())
    preds = np.concatenate(preds)
    
    # Align predictions with data
    df = X_selected.iloc[seq_len:].copy()
    df["obs_log_flux"] = y_test.iloc[seq_len:].values
    df["pred_log_flux"] = preds
    df["obs_flux"] = 10 ** df["obs_log_flux"]
    df["pred_flux"] = 10 ** df["pred_log_flux"]
    df.dropna(subset=["L_shell", "obs_flux", "pred_flux"], inplace=True)
    
    print("Data preparation complete.")
    return df


def create_flux_plot(df, energy_label, output_filename):
    print("Generating L–time flux map...")
    # Filter any nonpositive flux values
    df = df[(df["obs_flux"] > 0) & (df["pred_flux"] > 0)].copy()

    vmin, vmax = 1e2, 1e6
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True, sharey=True)
 
    scatter_style = { 's': 300, 'marker': '|', 'cmap': "turbo", 'norm': norm, 'rasterized': True }

    im1 = axes[0].scatter(df.index, df["L_shell"], c=df["obs_flux"], **scatter_style)
    axes[0].set_title(f"Observed RBSP MagEIS {energy_label} Electron Flux", fontsize=13)
    axes[0].set_ylabel("L-shell (Rₑ)")
    cbar1 = fig.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label(r"cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$")

    im2 = axes[1].scatter(df.index, df["L_shell"], c=df["pred_flux"], **scatter_style)
    axes[1].set_title(f"LSTM-Predicted RBSP MagEIS {energy_label} Electron Flux", fontsize=13)
    axes[1].set_ylabel("L-shell (Rₑ)")
    cbar2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label(r"cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$")

    storm_onset = pd.Timestamp("2017-03-01", tz="UTC")
    start_date = pd.Timestamp("2017-02-25", tz="UTC")
    end_date = pd.Timestamp("2017-03-25", tz="UTC")

    for ax in axes:
        ax.set_ylim(df["L_shell"].min() - 0.1, df["L_shell"].max() + 0.1)
        ax.set_xlim(start_date, end_date)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Flux plot saved to {output_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate flux maps for any channel")
    parser.add_argument("--channel", type=int, default=3, 
                        help="Channel number (3=54keV, 11=235keV, 14=597keV, 16=909keV)")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device number to use")
    
    args = parser.parse_args()
    
    # Validate channel
    if args.channel not in CHANNEL_MAP:
        raise ValueError(f"Invalid channel {args.channel}. Must be one of {list(CHANNEL_MAP.keys())}")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Get channel info
    channel_info = CHANNEL_MAP[args.channel]
    channel_name = channel_info['name']
    
    # Construct paths based on channel
    storm_X_path = f"dataset/preprocessed/{args.channel}/X_test_storm_norm.pkl"
    storm_y_path = f"dataset/preprocessed/{args.channel}/y_test_storm.pkl"
    mean_path = f"dataset/extracted/mageis_ch{args.channel}_input_avg.npy"
    std_path = f"dataset/extracted/mageis_ch{args.channel}_input_std.npy"
    raw_path = f"dataset/extracted/ch{args.channel}_data.pkl"
    
    # save to plots directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/L_time_fluxmap_LSTM_{channel_name}.png"
    
    # Get input columns from channel data
    positional_features = channel_info.get('positional_features', [])
    time_series_features = channel_info.get('time_series_features', [])
    
    if not positional_features or not time_series_features:
        raise ValueError(f"Channel {args.channel} does not have feature configuration set up in CHANNEL_MAP")
    
    input_cols = positional_features + [f"{prefix}_t_0" for prefix in time_series_features]
    
    # Get energy label
    energy_map = {
        3: "54 keV",
        11: "235 keV", 
        14: "597 keV",
        16: "909 keV"
    }
    energy_label = energy_map.get(args.channel, f"Channel {args.channel}")
    
    print(f"\n{'='*60}")
    print(f"Processing channel {args.channel} ({channel_name}, {energy_label})")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Try to load config.json from checkpoint directory to get exact training params
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        print(f"Loading training config from: {config_path}")
        import json
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        seq_length = training_config.get('seq_length', 1500)
        config_hidden_size = training_config.get('hidden_size', None)
        print(f"Found seq_length from training config: {seq_length}")
    else:
        print(f"Config file not found at {config_path}, using best_params")
        seq_length = channel_info.get('best_params', {}).get('seq_length', 1500)
        config_hidden_size = None
        print(f"Using seq_length from best_params: {seq_length}")
    
    # Load checkpoint 
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Try to infer hidden_size and seq_length from checkpoint
    # The model state dict has lstm.weight_hh_l0 with shape [hidden_size*4, hidden_size]
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Infer hidden size from LSTM weights
    lstm_weight_key = 'lstm.weight_hh_l0'
    if lstm_weight_key in state_dict:
        hidden_size = state_dict[lstm_weight_key].shape[1]
        print(f"Detected hidden_size from checkpoint weights: {hidden_size}")
        # Verify against config if available
        if config_hidden_size and config_hidden_size != hidden_size:
            print(f"Warning: Config has hidden_size={config_hidden_size}, but weights show {hidden_size}")
    else:
        # Fallback to config or best params
        if config_hidden_size:
            hidden_size = config_hidden_size
            print(f"Using hidden_size from config: {hidden_size}")
        else:
            hidden_size = channel_info.get('best_params', {}).get('hidden_size', 512)
            print(f"Using default hidden_size: {hidden_size}")
    
    # Setup config
    config = Config()
    config.channel = args.channel
    config.seq_length = seq_length
    config.hidden_size = hidden_size
    config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Load model
    ModelClass = get_model_class(args.channel)
    model = ModelClass(len(input_cols), config.hidden_size).to(config.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully! Using {ModelClass.__name__}")
    print(f"{'='*60}\n")

    df = load_and_prepare_data(config, model, args.channel, raw_path)
    
    # Filter by L-shell 
    L_SHELL_MIN = 2.6
    df = df[df["L_shell"] > L_SHELL_MIN].copy()
    print(f"Filtered data: keeping L-shell > {L_SHELL_MIN} ({len(df)} samples)")
    print(f"L-shell range: {df['L_shell'].min():.2f} - {df['L_shell'].max():.2f}\n")
    
    # Create plot
    create_flux_plot(df, energy_label, output_filename)
