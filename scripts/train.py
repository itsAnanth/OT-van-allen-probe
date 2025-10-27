import torch
import argparse
import os
import torch.nn as nn
import pickle as pkl
import gc
from tqdm import tqdm
from common.config import Config
from common.utils import set_random_seed, append_to_pickle, print_gpu_memory, save_checkpoint, load_checkpoint
from models.lstm import FluxLSTM
from scripts.dataset import load_data
from scripts.eval import evaluate
from dataclasses import fields



def tuning(config: Config):
    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
    params_to_tune_map = {
        'seq_length': range(500, 2500, 500),
        'hidden_size': [64, 128, 256, 512]
    }
    
    os.makedirs("tuning", exist_ok=True)  # works in os too

    
    if config.tune_param is None:
        params_to_tune = list(params_to_tune_map.items())
    else:
        assert config.tune_param in params_to_tune_map.keys(), "invalid parameter to tune"
        params_to_tune = [(config.tune_param, params_to_tune_map[config.tune_param])]
    
    
    for param, param_range in params_to_tune:
        
        print_gpu_memory(f"before tuning {param}")
        
        param_file_path = f"tuning/{param}.pkl"
        
        print(f"computing optimal value for {param}")
        for value in param_range:
            print(f"Training parameter {param} with value {value}")
            setattr(config, param, value)
            train_loader, val_loader, test_loader = load_data(config)
            input_size = next(iter(train_loader))[0].shape[2]
            
            model = FluxLSTM(input_size, config.hidden_size).to(device)
            metrics = train(model, train_loader, test_loader, config)
            
            metric = {
                'value': value,
                'metrics': metrics
            }
            
            append_to_pickle(param_file_path, metric)
            print(f"metric for {param} = {metric}")
            
            
            del train_loader, val_loader, test_loader, model
            torch.cuda.empty_cache()
            gc.collect()
            
            print_gpu_memory(f"after tuning {param}")
            
            print("-" * 50, end='\n')
        print("-" * 50, end="\n")

        
        
        print(f"Finished tuning {param}")




def train(model, train_loader, val_loader, config: Config):
    metrics = []
    

    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    if model is None:
        input_size = next(iter(train_loader))[0].shape[2]

        print("No model detected, defaulting to FluxLSTM")
        model = FluxLSTM(input_size, config.hidden_size).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    epoch_start = 0 if config.checkpoint_epoch == -1 else config.checkpoint_epoch
    print(f"Using device: {device}")

    
    if config.checkpoint_epoch != -1:
        print("loading checkpoint")
        model, optimizer = load_checkpoint(config.checkpoint_epoch, model, optimizer, config)

    for epoch in range(epoch_start, config.max_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in tqdm(train_loader, total=len(train_loader)):
            xb = xb.to(device)
            yb = yb.to(device)
            
            

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{config.max_epochs}, Loss: {epoch_loss:.4f}")
        
        r2_score = evaluate(model, val_loader, config)
        metric = {
            'loss': epoch_loss,
            'rscore': r2_score
        }
        metrics.append(metric)
        print(f"Validation r2 score: {r2_score}")
        
        if config.checkpoint:
            checkpoint_name = f"54kev_{epoch + 1}.pth"
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metric
            }
            print(f"Saving model checkpoint to {config.checkpoint_dir}/{checkpoint_name}")
            save_checkpoint(config.checkpoint_dir, checkpoint_name, checkpoint_data)
        
        
    return metrics

def config_from_args(cls, config: Config):
    arg_dict = vars(config)
    config_fields = {f.name for f in fields(cls)}
    filtered_args = {k: v for k, v in arg_dict.items() if k in config_fields}
    return cls(**filtered_args)

        
    

def parse_args():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default='dataset/preprocessed')
    parser.add_argument("--seq-length", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--load-from-checkpoint", action="store_true")
    parser.add_argument("--data-limit", action="store_true", help="slice data for testing")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--tune-param", type=str, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, default=-1)
    
    


    
    config, remaining_args = parser.parse_known_args()
    
    assert remaining_args == [], remaining_args
    
    if torch.cuda.is_available() and config.gpu not in range(0, torch.cuda.device_count()):
        raise ValueError(f"Invalid gpu id {config.gpu}")
    

    return config
    

    
    
if __name__ == "__main__":
    
    set_random_seed()
    config = parse_args()
    config = config_from_args(Config, config)
    
    if config.tune:
        config.checkpoint = False
        tuning(config)
    else:
        config.checkpoint = True
        train_loader, val_loader, test_loader = load_data(config)
        metrics = train(model=None, train_loader=train_loader, val_loader=val_loader, config=config)
        append_to_pickle('tuning/final.pkl', metrics)
    
