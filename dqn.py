import torch.nn as nn

class DQN(nn.Module):
    
    def __init__(self, n_steps):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(n_steps, n_steps * 2),
            # nn.BatchNorm1d(n_steps * 2),
            nn.Sigmoid(),
            
            nn.Linear(n_steps * 2, n_steps * 4),
            # nn.BatchNorm1d(n_steps * 4),
            nn.Sigmoid(),
            
            nn.Linear(n_steps * 4, n_steps * 4),
            # nn.BatchNorm1d(n_steps * 2),
            nn.Sigmoid(),
            
            nn.Linear(n_steps * 4 , n_steps * 2),
            nn.Sigmoid(),
            
            nn.Linear(n_steps * 2 , n_steps),
            nn.Sigmoid()
        )
        
        '''
        TO DO: consider other activations functions at the network head to output more extreme values, i.e. close to 0 or 1
        
            - Softmax is no good since it sums to 1, which is not what we want
            - Heaviside step ?
        '''
        
    def forward(self, x):
        # print("\tForward input has shape", x.shape)
        x = self.fc(x)
        return x
