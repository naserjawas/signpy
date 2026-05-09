import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Linear(in_features= hidden_dim * 2, 
                                    out_features=1
        )

    def forward(self, x):
        # x: (B, T, D)
        x, _ = self.lstm(x)         # (B, T, 2 * hidden_dim)
        x = self.classifier(x)      # (B, T, 1)
        out = x.squeeze(-1)         # (B, T)

        return out
