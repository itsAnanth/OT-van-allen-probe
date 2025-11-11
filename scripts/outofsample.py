import torch
import os
import argparse
import torch.nn as nn
from scripts.eval import evaluate
from scripts.dataset import load_data
from models.lstm import FluxLSTM
from common.config import Config, load_config
from common.utils import autodetect_device, set_random_seed



if __name__ == "__main__":
    set_random_seed(42)
    parser = argparse.ArgumentParser(description="Test Model on out o fsample data")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)


    args, remaining_args = parser.parse_known_args()

    assert remaining_args == [], remaining_args
    assert args.checkpoint_dir is not None, "Invalid model path"
    assert args.model is not None, "Invalid model name"



    config = load_config(args.checkpoint_dir)
    criterion = nn.MSELoss()

    train_loader, val_loader, test_loader = load_data(config)

    input_size = next(iter(train_loader))[0].shape[2]

    model = FluxLSTM(input_size, config.hidden_size).to(config.device)

    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, args.model), map_location=config.device)['model_state_dict'])

    print(evaluate(model, test_loader, config, criterion))


