
import torch
import torch.nn as nn
import torch.nn.functional as F

from CNN import Karnot
from LSTM import WoffMan

## Connecting all of the models 
class FullModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = Karnot()
        self.lstm = WoffMan(input_dim=452)


    def forward(self,frames):

        B,T,C,H,W = frames.shape

        features = []

        for t in range(T):

            f = self.cnn(frames[:,t])
            features.append(f)

        features = torch.stack(features,dim=1)

        rppg = self.lstm(features)

        return rppg



