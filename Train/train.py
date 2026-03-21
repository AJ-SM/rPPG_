import torch
import numpy as np 
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim
import cv2
import os 

from model.CNN import Karnot
from model.LSTM import WoffMan
from module.ImgtorPPG import process_rppg_pipeline
from Preprocess.loadFrameStream import sendVideoTrain
from Preprocess.video2Frame import createVideoBatchOfFive
from module.test import extract_data_from_video 
from Hooks.DepthHook import get_depth_from_wsl

# Paths
PATH_Attack = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\attack"
PATH_Real = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\real"
TEMP_IMG_PATH = r"D:\wsl_bridge_temp.jpg" # Temporary file for WSL Hook

epochs = 5
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = Karnot().to(device).float()
rnn_model = WoffMan(input_dim=1).to(device).float()
optimizer = optim.Adam(list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=lr)

loss_depth = nn.MSELoss()
loss_rppg = nn.MSELoss()

print(f"---- Initiating the Training for {epochs} epochs on {device} ----")

for epoch in range(epochs):
    # 1. Get Video

    videoPath, status = sendVideoTrain(PATH_Attack, PATH_Real)
    print("--- Video Featched ---")
    try:
        # 2. Extract rPPG ground truth for the whole video
        print("--- Initiatiating RPPG Analysis ---")
        frame_list, mask_list, embed_list, fps = extract_data_from_video(videoPath)
        print("--- Processing through Pipeline ---")
        rppg_res = process_rppg_pipeline(frame_list, mask_list, embed_list, fps)
        gt_pulse_full = rppg_res["results"]["POS"]["filtered_signal"]
        print("--- rPPG Completed BPM Aquired !!! ---")
    except Exception as e:
        print(f"Skipping video {videoPath} due to extraction error: {e}")
        continue

    idx = 0
    # 3. Process sliding windows of 5 frames
    for inputTensor, mid_pixels in createVideoBatchOfFive(videoPath):
        
        if status:
            print("--- Aqquired Real image ---")
            # --- THE FIX: Save pixels to disk so WSL can see the path ---
            cv2.imwrite(TEMP_IMG_PATH, mid_pixels)
            
            # Pass the PATH string to the hook
            print("--- Targetting Memory Hook for Depth Map ---")
            target_depth_np = get_depth_from_wsl(TEMP_IMG_PATH)
            
            # If WSL fails to find a face, skip this frame
            if target_depth_np is None:
                continue
            print("--- Depth Map Aquired ---")
            target_pulse = torch.tensor([gt_pulse_full[min(idx + 2, len(gt_pulse_full)-1)]],dtype=torch.float32).to(device)
        else:
            print("=== Aqquired Fake Image ===")
            # Attack logic: Surface is flat (zeros)
            target_depth_np = np.zeros((32, 32), dtype=np.float32)
            target_pulse = torch.tensor([gt_pulse_full[min(idx + 2, len(gt_pulse_full)-1)]],dtype=torch.float32).to(device)
            # target_pulse = torch.tensor([0.0]).to(device).float()

        # --- TENSOR PREPARATION ---
        print("--- Pushing the data via Pipeline ---")
        input_pt = torch.from_numpy(inputTensor).permute(2, 0, 1).unsqueeze(0).to(device).float()
        target_depth_pt = torch.from_numpy(target_depth_np).unsqueeze(0).unsqueeze(0).to(device).float()

        print("--- Initilizing Forward Pass ---")
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # CNN predicts (32, 32) depth and (32, 32) features
        pred_depth, raw_features = cnn_model(input_pt)
        
        # Flatten spatial features for LSTM (Global Average Pool)
        pooled = F.adaptive_avg_pool2d(raw_features, (1, 1)).view(1, 1, -1)
        pred_pulse = rnn_model(pooled)

        print("--- Calculating the loss ---")
        # --- LOSS & BACKWARD ---
        l_depth = loss_depth(pred_depth, target_depth_pt)
        print("Depth loss : ",l_depth)
        # l_rppg = loss_rppg(pred_pulse.squeeze(), target_pulse)
        print("Predicted PULSE : ",pred_pulse.view(-1),"Detected Pulse : ",target_pulse.view(-1))
        l_rppg= loss_rppg(pred_pulse.view(-1), target_pulse.view(-1))
        print("Loss rPPG: " ,l_rppg)

        total_loss = l_depth + l_rppg
        print("Total Loss : ",total_loss)

        print("--- Initiating Back-Prop ---")
        total_loss.backward()
        optimizer.step()

        idx += 1

    print(f"Epoch [{epoch}/{epochs}] Video: {os.path.basename(videoPath)} | Batch Loss: {total_loss.item():.4f}")

    # Optional: Save checkpoint every 10 videos
    if epoch % 10 == 0:
        torch.save(cnn_model.state_dict(), "cnn_checkpoint.pth")
        torch.save(rnn_model.state_dict(), "rnn_checkpoint.pth")

# Save the weights
torch.save(cnn_model.state_dict(), 'final_models/karnot_cnn_final.pth')
torch.save(rnn_model.state_dict(), 'final_models/woffman_lstm_final.pth')

print("Final models saved to final_models/ folder!")



















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



