import torch
import argparse
import os
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
from models.lstm import FluxLSTM
from scripts.dataset import load_data
from scripts.eval import evaluate
from pathlib import Path

"""

    TODO: LOGIC ERROR
    same train loader used for all the tuning hyper params
    dataset should change for params like seq_length
"""

def tuning(train_loader, args):
    input_size = next(iter(train_loader))[0].shape[2]
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    params_to_tune = [
        # ('seq_length', range(4000, 10000, 1000)),
        ('hidden_size', [64, 128, 256, 512])
    ]
    
    os.makedirs("tuning", exist_ok=True)  # works in os too

    
    
    for param, param_range in params_to_tune:
        
        param_metrics = []
        
        print(f"computing optimal value for {param}")
        for value in param_range:
            print(f"Training parameter {param} with value {value}")
            setattr(args, param, value)
            model = FluxLSTM(input_size, args.hidden_size).to(device)
            _, metrics = train(model, train_loader, test_loader, args)
            
            param_metrics.append({
                'value': value,
                'metrics': metrics
            })
            print("-" * 20)
        print(f"metric for {param} = {param_metrics}")
        print("-" * 50, end="\n")

        
        
        print(f"Writing metrics to tuning/{param}")
        with open(f"tuning/{param}.pkl", 'wb') as f:
            pkl.dump(param_metrics, f)
        print("Finished writing\n")
            



def train(model, train_loader, val_loader, args):
    metrics = []
    

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        print("No model detected, defaulting to FluxLSTM")
        model = FluxLSTM(input_size, args.hidden_size).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")


    for epoch in range(args.max_epochs):
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
        print(f"Epoch {epoch+1}/{args.max_epochs}, Loss: {epoch_loss:.4f}")
        
        r2_score = evaluate(model, val_loader, args)
        metrics.append({
            'loss': epoch_loss,
            'rscore': r2_score
        })
        print(f"Validation r2 score: {r2_score}")
        
        
    return model, metrics
        
    

def parse_args():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--data_path", type=str, default='../processed_data')
    parser.add_argument("--seq_length", type=int, default=6000) # 6000 after tuning
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
    
    if torch.cuda.is_available() and args.gpu not in range(0, torch.cuda.device_count()):
        raise ValueError(f"Invalid gpu id {args.gpu}")
    
    if args.channel == 3:
        args.positional_features = ["ED_R_OP77Q_intxt", "ED_MLAT_OP77Q_intxt", "ED_MLT_OP77Q_intxt_sin", "ED_MLT_OP77Q_intxt_cos"]
        
    return args
    

    
    
if __name__ == "__main__":
    args = parse_args()
    train_loader, val_loader, test_loader = load_data(args)
    
    if args.tune:
        tuning(train_loader, args)
    else:
        train(train_loader, args)
    
