import torch
import torch.nn as nn
import torch.nn.functional as F


class Karnot(nn.Module):

    def __init__(self):
        super(Karnot, self).__init__()

        # -------- Block 1 --------
        self.convB11 = nn.Conv2d(6, 64, 3, padding=1)
        
        self.convB12 = nn.Conv2d(64, 128, 3, padding=1)
        self.convB13 = nn.Conv2d(128, 196, 3, padding=1)

        self.bnB1 = nn.BatchNorm2d(196)
        self.poolB14 = nn.MaxPool2d(2, 2)

        # -------- Block 2 --------
        self.convB21 = nn.Conv2d(196, 128, 3, padding=1)
        self.convB22 = nn.Conv2d(128, 196, 3, padding=1)
        self.convB23 = nn.Conv2d(196, 128, 3, padding=1)

        self.bnB2 = nn.BatchNorm2d(128)
        self.poolB24 = nn.MaxPool2d(2, 2)

        # -------- Block 3 --------
        self.convB31 = nn.Conv2d(128, 128, 3, padding=1)
        self.convB32 = nn.Conv2d(128, 196, 3, padding=1)
        self.convB33 = nn.Conv2d(196, 128, 3, padding=1)

        self.bnB3 = nn.BatchNorm2d(128)
        self.poolB34 = nn.MaxPool2d(2, 2)

        # -------- Feature Branch (FIXED) --------
        self.Fconv1 = nn.Conv2d(452, 256, 3, padding=1)
        self.Fconv2 = nn.Conv2d(256, 128, 3, padding=1)
        # ❌ removed collapsing to 1 channel

        # -------- Depth Branch --------
        self.DEconv1 = nn.Conv2d(452, 128, 3, padding=1)
        self.DEconv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.DEconv3 = nn.Conv2d(64, 1, 3, padding=1)

        self.elu = nn.ELU()

    def forward(self, x):

        # -------- Block 1 --------
        x1 = self.poolB14(self.elu(self.bnB1(self.convB13(self.convB12(self.convB11(x))))))

        # -------- Block 2 --------
        x2 = self.poolB24(self.elu(self.bnB2(self.convB23(self.convB22(self.convB21(x1))))))

        # -------- Block 3 --------
        x3 = self.poolB34(self.elu(self.bnB3(self.convB33(self.convB32(self.convB31(x2))))))

        # -------- Resize --------
        x1 = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)

        # -------- Concatenate --------
        x = torch.cat([x1, x2, x3], dim=1)  # (B, 452, H, W)

        # -------- Depth Branch --------
        xo = self.elu(self.DEconv1(x))
        xo = self.elu(self.DEconv2(xo))
        depthMap = self.DEconv3(xo)  # (B,1,H,W)

        # -------- Feature Branch (FIXED) --------
        yo = self.elu(self.Fconv1(x))
        features = self.elu(self.Fconv2(yo))  # (B,128,H,W)

        return depthMap, features