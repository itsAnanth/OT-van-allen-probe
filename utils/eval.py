import torch
import numpy as np
from sklearn.metrics import r2_score


def eval(model, test_loader):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)

            # Ensure both preds and targets are 1D numpy arrays
            all_preds.append(outputs.view(-1).cpu().numpy())
            all_targets.append(y_batch.view(-1).cpu().numpy())

    # Concatenate along axis 0
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # R² score
    r2 = r2_score(all_targets, all_preds)
    print("R² Score:", r2)

    return r2
