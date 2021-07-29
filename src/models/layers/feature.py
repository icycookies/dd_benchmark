import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_features, hidden_sizes, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(num_features, hidden_sizes[0])] +
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x