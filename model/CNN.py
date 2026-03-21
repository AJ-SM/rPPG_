import torch
import torch.nn as nn
import torch.nn.functional as F


class Karnot(nn.Module):

    def __init__(self):
        super(Karnot,self).__init__()

        # -------- Block 1 --------
        self.convB11 = nn.Conv2d(6,64,3,padding=1)
        self.convB12 = nn.Conv2d(64,128,3,padding=1)
        self.convB13 = nn.Conv2d(128,196,3,padding=1)

        self.bnB1 = nn.BatchNorm2d(196)
        self.poolB14 = nn.MaxPool2d(2,2)

        # -------- Block 2 --------
        self.convB21 = nn.Conv2d(196,128,3,padding=1)
        self.convB22 = nn.Conv2d(128,196,3,padding=1)
        self.convB23 = nn.Conv2d(196,128,3,padding=1)

        self.bnB2 = nn.BatchNorm2d(128)
        self.poolB24 = nn.MaxPool2d(2,2)

        # -------- Block 3 --------
        self.convB31 = nn.Conv2d(128,128,3,padding=1)
        self.convB32 = nn.Conv2d(128,196,3,padding=1)
        self.convB33 = nn.Conv2d(196,128,3,padding=1)

        self.bnB3 = nn.BatchNorm2d(128)
        self.poolB34 = nn.MaxPool2d(2,2)

        # After Concatenation 

        # Feature branch 
        self.Fconv1 = nn.Conv2d(452,128,3,padding=1)
        self.Fconv2 = nn.Conv2d(128,3,3,padding=1)
        self.Fconv3 = nn.Conv2d(3,1,3,padding=1)

        # Depth branch 
        self.DEconv1 = nn.Conv2d(452,128,3,padding=1)
        self.DEconv2 = nn.Conv2d(128,64,3,padding=1)
        self.DEconv3 = nn.Conv2d(64,1,3,padding=1)

        self.elu = nn.ELU()


    def forward(self,x):

        # Block 1 
        x1 = self.convB11(x)
        x1 = self.convB12(x1)
        x1 = self.convB13(x1)
        x1 = self.bnB1(x1)
        x1 = self.elu(x1)
        x1 = self.poolB14(x1)

        # Block 2 
        x2 = self.convB21(x1)
        x2 = self.convB22(x2)
        x2 = self.convB23(x2)
        x2 = self.bnB2(x2)
        x2 = self.elu(x2)
        x2 = self.poolB24(x2)

        # Block 3 
        x3 = self.convB31(x2)
        x3 = self.convB32(x3)
        x3 = self.convB33(x3)
        x3 = self.bnB3(x3)
        x3 = self.elu(x3)
        x3 = self.poolB34(x3)

        #  Resize for fusion 
        x1 = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate 
        x = torch.cat([x1, x2, x3], dim=1)   

        # Depth Branch 
        xo = self.elu(self.DEconv1(x))
        xo = self.elu(self.DEconv2(xo))
        depthMap = self.DEconv3(xo)          # (B,1,H,W)

        # Feature Branch 
        yo = self.elu(self.Fconv1(x))
        yo = self.elu(self.Fconv2(yo))
        features = self.Fconv3(yo)           # (B,1,H,W)

        return depthMap, features