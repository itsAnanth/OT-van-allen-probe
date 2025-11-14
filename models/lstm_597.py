import torch
import torch.nn as nn

class FluxLSTM_597keV(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(FluxLSTM_597keV, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bias=True
        )
        # MLP head: [hidden_size → 432 → 114 → 93 → 12 → 1]
        self.fc = nn.Sequential(
            # Layer 0
            nn.Linear(hidden_size, 432),
            nn.BatchNorm1d(432),
            nn.ReLU(),
            nn.Dropout(p=0.011),

            # Layer 1
            nn.Linear(432, 114),
            nn.BatchNorm1d(114),
            nn.ReLU(),
            nn.Dropout(p=0.092),

            # Layer 2 (dropout = 0.0 per Table S1)
            nn.Linear(114, 93),
            nn.BatchNorm1d(93),
            nn.ReLU(),

            # Layer 3 (dropout = 0.0)
            nn.Linear(93, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),

            # Output layer — no BN, no activation
            nn.Linear(12, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        out = out[:, -1, :]            # (batch, hidden_size)
        out = self.fc(out)             # (batch, 1)
        return out.squeeze(-1)         # (batch,)