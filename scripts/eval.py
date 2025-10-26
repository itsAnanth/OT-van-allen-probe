import torch
import numpy as np
from sklearn.metrics import r2_score

# evaluate model

def evaluate(model, loader, args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)

            # Ensure both preds and targets are 1D numpy arrays
            all_preds.append(outputs.view(-1).cpu().numpy())
            all_targets.append(y_batch.view(-1).cpu().numpy())

    # Concatenate along axis 0
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # R² score
    r2 = r2_score(all_targets, all_preds)
        
    model.train()

    return r2
