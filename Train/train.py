
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FullModel import FullModel



model = FullModel()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.L1Loss()

epochs = 10000

for epoch in range(epochs):

    for video,rppg_gt in loader:

        video = video
        rppg_gt = rppg_gt

        pred = model(video)

        loss = criterion(pred,rppg_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss:",loss.item())