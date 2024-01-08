import torch.nn as nn
from envs import NUM_ACTIONS, NUM_INPUTS


class PokerNet(nn.Module):
    def __init__(self, n_inputs=NUM_INPUTS, n_actions=NUM_ACTIONS):
        super(PokerNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_actions)

    def forward(self, x):
        return self.fc1(x)
