import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(
                in_channels=input_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            ))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(in_channels=hidden_dim,
                                    out_channels=1,
                                    kernel_size=1
        )


    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)   # (B, D, T)
        x = self.network(x)
        x = self.classifier(x)
        out = x.transpose(1, 2) # (B, T, 1)

        return torch.sigmoid(out)
