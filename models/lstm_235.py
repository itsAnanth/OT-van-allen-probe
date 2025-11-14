import torch
import torch.nn as nn

class FluxLSTM_235keV(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(FluxLSTM_235keV, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            bias=True
        )
        # MLP head: [hidden_size → 412 → 105 → 116 → 128 → 1]
        self.fc = nn.Sequential(
            # Layer 0
            nn.Linear(hidden_size, 412),
            nn.BatchNorm1d(412),
            nn.ReLU(),
            nn.Dropout(p=0.368),

            # Layer 1
            nn.Linear(412, 105),
            nn.BatchNorm1d(105),
            nn.ReLU(),
            nn.Dropout(p=0.091),

            # Layer 2 (no dropout per Table S1: dropout=0.0 for layers 2 & 3)
            nn.Linear(105, 116),
            nn.BatchNorm1d(116),
            nn.ReLU(),

            # Layer 3
            nn.Linear(116, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Output layer — no BN, no activation (regression target)
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)        # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]          # (batch, hidden_size)
        out = self.fc(out)           # (batch, 1)
        return out.squeeze(-1)       # (batch,) — matches scalar target log10(flux+1)