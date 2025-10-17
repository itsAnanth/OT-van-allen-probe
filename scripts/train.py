from models.lstm import FluxLSTM
from scripts.data import get_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--data_path", type=str, default='../processed_data')
    parser.add_argument("--seq_length", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    return parser, args

def train(loader, args):
    input_size = X.shape[2]  # number of features (excluding target flux)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FluxLSTM(input_size, args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")
        
    

def main():
    parser, args = parse_args()
    
    
if __name__ == "__main__":
    main()
    
