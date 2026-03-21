import torch
import torch.nn as nn
import torch.nn.functional as F



class WoffMan(nn.Module):

    def __init__(self, input_dim):
        super(WoffMan,self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(100,1)


    def forward(self,x):
        # x: (B, T, input_dim)qq

        out,_ = self.lstm(x)              # (B, T, 100)

        rppg = torch.tanh(self.fc(out))  # (B, T, 1)

        return rppg
