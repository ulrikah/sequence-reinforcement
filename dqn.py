import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.Sigmoid(),
            nn.Linear(in_dim * 2, in_dim * 4),
            nn.Sigmoid(),
            nn.Linear(in_dim * 4, in_dim * 4),
            nn.Sigmoid(),
            nn.Linear(in_dim * 4, in_dim * 2),
            nn.Sigmoid(),
            nn.Linear(in_dim * 2, out_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.fc(x)
        return x
