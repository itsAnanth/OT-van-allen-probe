import torch
import torch.nn as nn


class MLP54:
    def __init__(self, hidden_size=64):
        # hidden size will be the input from lstm
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
        return self.fc(x)
