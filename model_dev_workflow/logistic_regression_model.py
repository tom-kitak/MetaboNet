import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, l1_value, device):
        super(LogisticRegressionModel, self).__init__()
        self.device = device
        self.l1_value = l1_value
        self.linear = nn.Linear(input_dim, 1).to(self.device)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))

    def l1_regularization(self):
        return self.l1_value * torch.sum(torch.abs(self.linear.weight))
