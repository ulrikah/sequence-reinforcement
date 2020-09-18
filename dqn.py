import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.fc(x)
        return x
