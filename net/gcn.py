import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x1 = a @ self.linear1(x)
        x1 = self.relu(x1)
        x2 = a @ self.linear2(x1)
        return x2


class Classifier(nn.Module):
    def __init__(self, i_dim, o_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(i_dim, o_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y
