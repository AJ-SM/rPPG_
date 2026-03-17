
import torch
import torch.nn as nn
import torch.nn.functional as F


class Karnot(nn.Module):

    def __init__(self):
        super(Karnot,self).__init__()

        # Block 1
        self.convB11 = nn.Conv2d(6,64,3,padding=1)
        self.convB12 = nn.Conv2d(64,128,3,padding=1)
        self.convB13 = nn.Conv2d(128,196,3,padding=1)

        self.bnB1 = nn.BatchNorm2d(196)
        self.poolB14 = nn.MaxPool2d(2,2)

        # Block 2
        self.convB21 = nn.Conv2d(196,128,3,padding=1)
        self.convB22 = nn.Conv2d(128,196,3,padding=1)
        self.convB23 = nn.Conv2d(196,128,3,padding=1)

        self.bnB2 = nn.BatchNorm2d(128)
        self.poolB24 = nn.MaxPool2d(2,2)

        # Block 3
        self.convB31 = nn.Conv2d(128,128,3,padding=1)
        self.convB32 = nn.Conv2d(128,196,3,padding=1)
        self.convB33 = nn.Conv2d(196,128,3,padding=1)

        self.bnB3 = nn.BatchNorm2d(128)
        self.poolB34 = nn.MaxPool2d(2,2)

        self.relu = nn.ReLU()

        # reduce feature size for LSTM
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self,x):

        # Block 1
        x1 = self.convB11(x)
        x1 = self.convB12(x1)
        x1 = self.convB13(x1)
        x1 = self.bnB1(x1)
        x1 = self.relu(x1)
        x1 = self.poolB14(x1)

        # Block 2
        x2 = self.convB21(x1)
        x2 = self.convB22(x2)
        x2 = self.convB23(x2)
        x2 = self.bnB2(x2)
        x2 = self.relu(x2)
        x2 = self.poolB24(x2)

        # Block 3
        x3 = self.convB31(x2)
        x3 = self.convB32(x3)
        x3 = self.convB33(x3)
        x3 = self.bnB3(x3)
        x3 = self.relu(x3)
        x3 = self.poolB34(x3)

       
        x1 = F.interpolate(x1,size=x3.shape[2:])
        x2 = F.interpolate(x2,size=x3.shape[2:])

      
        x = torch.cat([x1,x2,x3],dim=1)

        # global pooling
        x = self.global_pool(x)

        # flatten
        x = torch.flatten(x,1)

        return x
    









