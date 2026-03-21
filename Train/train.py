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

# --- NEW: Non-rigid Registration Logic ---
def non_rigid_registration(feature_map, pred_depth, threshold=0.1):
    """
    Implements the masking logic from the paper[cite: 241, 244].
    feature_map: (B, C, 32, 32)
    pred_depth: (B, 1, 32, 32)
    """
    # Create visibility mask V (binary indicator) [cite: 242]
    mask = (pred_depth > threshold).float()
    
    # Apply mask to features (U = T ⊙ V) [cite: 244]
    # This removes background and weakens activations for spoofs [cite: 253, 256]
    registered_features = feature_map * mask
    return registered_features

# Paths
PATH_Attack = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\attack"
PATH_Real = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\real"
TEMP_IMG_PATH = r"D:\wsl_bridge_temp.jpg" 

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
    videoPath, status = sendVideoTrain(PATH_Attack, PATH_Real)
    
    try:
        frame_list, mask_list, embed_list, fps = extract_data_from_video(videoPath)
        rppg_res = process_rppg_pipeline(frame_list, mask_list, embed_list, fps)
        gt_pulse_full = rppg_res["results"]["POS"]["filtered_signal"]
    except Exception as e:
        print(f"Skipping video due to error: {e}")
        continue

    idx = 0
    for inputTensor, mid_pixels in createVideoBatchOfFive(videoPath):
        if status:
            cv2.imwrite(TEMP_IMG_PATH, mid_pixels)
            target_depth_np = get_depth_from_wsl(TEMP_IMG_PATH)
            if target_depth_np is None: continue
            target_pulse = torch.tensor([gt_pulse_full[min(idx + 2, len(gt_pulse_full)-1)]], dtype=torch.float32).to(device)
        else:
            target_depth_np = np.zeros((32, 32), dtype=np.float32)
            target_pulse = torch.tensor([0.0]).to(device).float()

        input_pt = torch.from_numpy(inputTensor).permute(2, 0, 1).unsqueeze(0).to(device).float()
        target_depth_pt = torch.from_numpy(target_depth_np).unsqueeze(0).unsqueeze(0).to(device).float()

        optimizer.zero_grad()

        # --- FORWARD PASS WITH REGISTRATION ---
        # 1. CNN generates depth map and raw spatial features
        pred_depth, raw_features = cnn_model(input_pt)
        
        # 2. Apply Non-rigid Registration (Masking) [cite: 31, 237]
        # This replaces the simple Global Average Pool you had before
        registered_features = non_rigid_registration(raw_features, pred_depth)
        
        # 3. Process registered features for RNN
        # Flatten the spatial dimensions of the MASKED features [cite: 252]
        pooled = F.adaptive_avg_pool2d(registered_features, (1, 1)).view(1, 1, -1)
        pred_pulse = rnn_model(pooled)

        # --- LOSS & BACKWARD ---
        l_depth = loss_depth(pred_depth, target_depth_pt)
        l_rppg = loss_rppg(pred_pulse.view(-1), target_pulse.view(-1))
        
        # Backpropagation updates both CNN and RNN parts in an end-to-end manner [cite: 230]
        total_loss = l_depth + l_rppg
        total_loss.backward()
        optimizer.step()
        idx += 1

    print(f"Epoch [{epoch}/{epochs}] Video: {os.path.basename(videoPath)} | Batch Loss: {total_loss.item():.4f}")

# Save final weights
torch.save(cnn_model.state_dict(), 'final_models/karnot_cnn_final.pth')
torch.save(rnn_model.state_dict(), 'final_models/woffman_lstm_final.pth')

















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



