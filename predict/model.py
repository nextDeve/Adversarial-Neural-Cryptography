import torch
import torch.nn as nn
import torch.nn.functional as F


class Alice(nn.Module):
    def __init__(self, config):
        super(Alice, self).__init__()
        self.hidden = config['alice']['hidden']
        self.depth = config['alice']['depth']
        self.cipher = config['cipher']
        self.input = nn.Linear(config['plain'] + config['key'], self.hidden)
        self.mlp = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth - 1)]
        )
        self.output = nn.Linear(self.hidden, self.cipher)

    def forward(self, p, k):
        input = torch.cat((p, k), dim=-1)
        hidden = F.relu(self.input(input))
        for idx, layer in enumerate(self.mlp):
            hidden = F.relu(hidden + layer(hidden))

        output = torch.tanh(self.output(hidden))
        return output


class Bob(nn.Module):
    def __init__(self, config):
        super(Bob, self).__init__()
        self.hidden = config['bob']['hidden']
        self.depth = config['bob']['depth']
        self.cipher = config['cipher']
        self.input = nn.Linear(self.cipher + config['key'], self.hidden)
        self.mlp = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth - 1)]
        )
        self.output = nn.Linear(self.hidden, config['plain'])

    def forward(self, c, k):
        input = torch.cat((c, k), dim=-1)
        hidden = F.relu(self.input(input))
        for idx, layer in enumerate(self.mlp):
            hidden = F.relu(hidden + layer(hidden))

        output = torch.tanh(self.output(hidden))
        return output


class Eve(nn.Module):
    def __init__(self, config):
        super(Eve, self).__init__()
        self.hidden = config['eve']['hidden']
        self.depth = config['eve']['depth']
        self.cipher = config['cipher']
        self.input = nn.Linear(self.cipher, self.hidden)
        self.mlp = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth - 1)]
        )
        self.output = nn.Linear(self.hidden, config['plain'])

    def forward(self, c):
        hidden = F.relu(self.input(c))
        for idx, layer in enumerate(self.mlp):
            hidden = F.relu(hidden + layer(hidden))

        output = torch.tanh(self.output(hidden))
        return output
