import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(101, 80)
        self.layer2 = nn.Linear(81, 100)
    def forward(self, x, device):
        x_shape = x.size(dim=0)
        one_units = torch.ones([x_shape,1]).to(device)

        output = torch.cat([x, one_units],1)
        output = self.layer1(output)
        hidden = torch.sigmoid(output)

        output = torch.cat([hidden, one_units], 1)
        output = self.layer2(output)
        output = torch.sigmoid(output)
        return output, hidden

class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(81, 100)
        self.layer2 = nn.Linear(101, 80)
    def forward(self, x, device):
        x_shape = x.size(dim=0)
        one_units = torch.ones([x_shape,1]).to(device)

        output = torch.cat([x, one_units],1)
        output = self.layer1(output)
        hidden = torch.sigmoid(output)

        output = torch.cat([hidden, one_units], 1)
        output = self.layer2(output)
        output = torch.sigmoid(output)
        return output, hidden
