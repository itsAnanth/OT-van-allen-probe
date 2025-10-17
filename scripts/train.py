import torch
import torch.nn as nn
from models.lstm import FluxLSTM
from scripts.dataset import load_data
from common.utils import parse_args



def train(train_loader, args):
    metrics = []
    
    input_size = next(iter(train_loader))[0].shape[2]  # number of features (excluding target flux)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FluxLSTM(input_size, args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")


    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
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
        
    return model
        
    

def main():
    parser, args = parse_args()
    
    train_loader, val_loader, test_loader = load_data(args)
    
    train(train_loader, args)
    
    
if __name__ == "__main__":
    main()
    
