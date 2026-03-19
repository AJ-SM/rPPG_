


import torch
import numpy as np 
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim
from model.CNN import Karnot
from model.LSTM import WoffMan
# from model.FullModel import FullModel
from module.ImgtorPPG import process_rppg_pipeline
from Preprocess.loadFrameStream import sendVideoTrain
from Preprocess.video2Frame import createVideoBatchOfFive
from module.test import extract_data_from_video 
from Hooks.DepthHook import get_depth_from_wsl
import os 



PATH_Attack = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\attack"
PATH_Real = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\real"


epochs =1000
lr = 1e-4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = Karnot().to(device)
rnn_model = WoffMan(input_dim=1).to(device) # Matches your feature branch output channel
optimizer = optim.Adam(list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=lr)


loss_depth = nn.MSELoss()
loss_rppg = nn.MSELoss()
loss_cls = nn.BCELoss()



for epoch in range(epochs):
    # Fix: Pass both paths
    videoPath, status = sendVideoTrain(PATH_Attack, PATH_Real)

    # Extract rPPG for the video
    frame, mask, embed, fps = extract_data_from_video(videoPath)
    rppg_res = process_rppg_pipeline(frame, mask, embed, fps)
    gt_pulse_full = rppg_res["results"]["POS"]["filtered_signal"]

    idx = 0
    # Process the sliding windows
    for inputTensor, mid in createVideoBatchOfFive(videoPath):
        if status:
            target_depth_np = get_depth_from_wsl(mid)
            target_label = torch.tensor([1.0]).to(device)
            # Ensure index doesn't go out of bounds
            pulse_idx = min(idx + 2, len(gt_pulse_full) - 1)
            target_pulse = torch.tensor([gt_pulse_full[pulse_idx]]).to(device)
        else:
            target_depth_np = np.zeros((32, 32), dtype=np.float32)
            target_label = torch.tensor([0.0]).to(device)
            target_pulse = torch.tensor([0.0]).to(device)

        # Prepare Tensors
        input_ = torch.from_numpy(inputTensor).permute(2, 0, 1).unsqueeze(0).to(device)
        target_depth_ = torch.from_numpy(target_depth_np).unsqueeze(0).unsqueeze(0).to(device)

        optimizer.zero_grad()

        # Forward
        pred_depth, raw_features = cnn_model(input_)
        
        # This matches your Karnot output (B, 1, 32, 32)
        pooled = F.adaptive_avg_pool2d(raw_features, (1, 1)).view(1, 1, -1)
        pred_pulse = rnn_model(pooled)

        # Loss
        loss_d = loss_depth(pred_depth, target_depth_)
        loss_p = loss_rppg(pred_pulse.squeeze(), target_pulse)

        total_loss = loss_d + loss_p
        
        # Backward & Update
        total_loss.backward()
        optimizer.step()

        idx += 1

    print(f"Epoch [{epoch}/{epochs}] Video: {os.path.basename(videoPath)} | Final Batch Loss: {total_loss.item():.4f}")




























# OLD CODE 
# import torch
# import numpy as np 
# import torch.nn.functional as F
# import torch.nn as nn 
# import torch.optim as optim
# from model.CNN import Karnot
# from model.LSTM import WoffMan
# from model.FullModel import FullModel
# from module.ImgtorPPG import process_rppg_pipeline
# from Preprocess.loadFrameStream import sendVideoTrain
# from Preprocess.video2Frame import createVideoBatchOfFive
# from module.test import extract_data_from_video 
# from Hooks.DepthHook import get_depth_from_wsl
# import os 


# epochs = 69
# lr = 1e-4 

# PATH_Attack = ""
# PATH_Real = ""

# cnn_model = Karnot()
# rnn_model = WoffMan()
# optimizer = optim.Adam(list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=lr)


# loss_depth = nn.MSELoss()
# loss_rppg = nn.MSELoss()
# loss_cls = nn.BCELoss()



# print(f"---- Initiating the Trainnig for {epochs} epochs ----")
# for epoch in range(epochs):

#     videoPath,status = sendVideoTrain(PATH_Attack,PATH_Real)

# # Maybe Potential Error : 
#     frame,mask,embed,fps = extract_data_from_video(videoPath)
#     rppg = process_rppg_pipeline(frame,mask,embed,fps)
#     gt_pulse_full=rppg["results"]["POS"]["filtered_signal"]

#     idx=0
#     for inputTensor, mid in createVideoBatchOfFive(videoPath):
#         if status:
#             target_depth =get_depth_from_wsl(mid)
#             target_label = torch.tensor([1.0])

#             target_pulse = torch.tensor([gt_pulse_full[idx+2]])

#         else:
#             target_depth = np.zeros((32,32),dtype=np.float32)
#             target_label = torch.tensor([0.0])
#             target_pulse = torch.tensor([0.0])

        
#         input_= torch.from_numpy(inputTensor).permute(2,0,1).unsqueeze(0)
#         target_depth_ = torch.from_numpy(target_depth).unsqueeze(0)

#         optimizer.zero_grad()


#         pred_depth,raw_features = cnn_model(input_)
#         pooled = F.adaptive_avg_pool2d(raw_features, (1, 1)).view(1, 1, -1)



#         pred_pules = rnn_model(pooled)

#         loss_d = loss_depth(pred_depth,target_depth)
#         loss_p = loss_rppg(pred_pules.squeeze(),target_pulse)


#         total_loss = loss_d+loss_p
#         total_loss.backword()
#         optimizer.step()

#         idx+=1
#     print(f"Epoch [{epoch}/{epochs}] Video: {os.path.basename(videoPath)} | Loss: {total_loss.item():.4f}")

        

#     # inputTensor,_ = createVideoBatchOfFive(videoPath)



