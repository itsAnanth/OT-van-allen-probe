from common.utils import parse_args
from scripts.dataset import load_data


def tuning():
    
    train_loader, val_loader, test_loader = load_data(args)
    
    params_to_tune = [
        ('seq_length', range(4000, 10000, 1000))
    ]
    
    
    for param, param_range in params_to_tune:
        
        print(f"computing optimal value for {param}")
        for value in param_range:


    # TODO:
