import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, p_dropout=0.5):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.linear1(self.dropout(x)))
        y = self.sigmoid(self.linear2(self.dropout(h)))
        return y
