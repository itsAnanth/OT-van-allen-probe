# 2. Define LSTM model
class FluxLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(FluxLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 291),
            nn.BatchNorm1d(291),
            nn.ReLU(),
            nn.Dropout(p=0.211),

            nn.Linear(291, 119),
            nn.BatchNorm1d(119),
            nn.ReLU(),
            nn.Dropout(p=0.385),

            nn.Linear(119, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Linear(4, 29),
            nn.BatchNorm1d(29),
            nn.ReLU(),

            nn.Linear(29, 1)   # final output, no BN/activation
        )



    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take only last time step's output
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()
