import torch
import numpy as np 
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim
import cv2
import os 
import random

from model.CNN import Karnot
from model.LSTM import WoffMan
from module.ImgtorPPG import process_rppg_pipeline
from Preprocess.video2Frame import createVideoBatchOfFive
from module.test import extract_data_from_video 
from Hooks.DepthHook import get_depth_from_wsl


# ---------------- NON-RIGID REGISTRATION ---------------- #
def non_rigid_registration(feature_map, pred_depth, threshold=0.1):
    mask = (pred_depth > threshold).float()
    return feature_map * mask


# ---------------- PATHS ---------------- #
PATH_Attack = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\attack"
PATH_Real = r"D:\Storeage-1\Main\ML-Model\data_set\DS\vvx\MSU-MFSD-Publish\scene01\real"
TEMP_IMG_PATH = r"D:\wsl_bridge_temp.jpg" 


# ---------------- HYPERPARAMS ---------------- #
epochs = 10
lr = 1e-4
lamb = 1.0
N_FFT = 64   # 🔥 FIXED FFT SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = Karnot().to(device).float()
rnn_model = WoffMan(input_dim=128, fs=35).to(device).float()

optimizer = optim.Adam(
    list(cnn_model.parameters()) + list(rnn_model.parameters()),
    lr=lr
)

loss_depth = nn.MSELoss()

print(f"---- Training for {epochs} epochs on {device} ----")

vid_count =0 
# ================= TRAIN LOOP ================= #
for epoch in range(epochs):

    attack_videos = [(os.path.join(PATH_Attack, v), False) for v in os.listdir(PATH_Attack)]
    real_videos = [(os.path.join(PATH_Real, v), True) for v in os.listdir(PATH_Real)]

    video_list = attack_videos + real_videos

    random.shuffle(video_list)

    epoch_loss = 0
    count = 0

    for videoPath, status in video_list:

        try:
            frame_list, mask_list, embed_list, fps = extract_data_from_video(videoPath)

            rppg_res = process_rppg_pipeline(frame_list, mask_list, embed_list, fps)
            gt_pulse = rppg_res["results"]["POS"]["filtered_signal"]

        except Exception as e:
            print(f"Skipping video: {e}")
            continue

        for inputTensor, mid_pixels in createVideoBatchOfFive(videoPath):

            # ---------------- INPUT ---------------- #
            input_pt = torch.from_numpy(inputTensor.copy())\
                .permute(0, 3, 1, 2)\
                .unsqueeze(0)\
                .to(device).float()

            B, T, C, H, W = input_pt.shape

            # ---------------- TARGET DEPTH ---------------- #
            if status:
                cv2.imwrite(TEMP_IMG_PATH, mid_pixels)
                target_depth_np = get_depth_from_wsl(TEMP_IMG_PATH)
                if target_depth_np is None:
                    continue
            else:
                target_depth_np = np.zeros((32, 32), dtype=np.float32)

            target_depth_pt = torch.from_numpy(target_depth_np.copy())\
                .unsqueeze(0).unsqueeze(0).to(device).float()

            optimizer.zero_grad()

            # ---------------- CNN ---------------- #
            pred_depth, raw_features = cnn_model(input_pt.squeeze(0))

            # ---------------- MASKING ---------------- #
            registered_features = non_rigid_registration(raw_features, pred_depth)

            # ---------------- POOL ---------------- #
            pooled = F.adaptive_avg_pool2d(
                registered_features, (1, 1)
            ).view(1, T, -1)

            # ---------------- LSTM ---------------- #
            pred_pulse = rnn_model(pooled)

            # ---------------- TARGET RPPG ---------------- #
            if status:
                gt_tensor = torch.from_numpy(
                    np.ascontiguousarray(gt_pulse)
                ).float().to(device)

                gt_tensor = (gt_tensor - gt_tensor.mean()) / (gt_tensor.std() + 1e-6)

                target_fft = torch.fft.rfft(gt_tensor, n=N_FFT)
                target_f = torch.abs(target_fft)

                freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / fps).to(device)
                mask = (freqs >= 0.7) & (freqs <= 4.0)

                target_f = target_f * mask
                target_pulse = F.normalize(target_f, p=2, dim=-1).unsqueeze(0)

            else:
                target_pulse = torch.zeros_like(pred_pulse).to(device)

            # ---------------- LOSS ---------------- #
            mid = T // 2
            l_depth = loss_depth(pred_depth[mid].unsqueeze(0), target_depth_pt)

            l_rppg = F.l1_loss(pred_pulse, target_pulse)

            total_loss = lamb * l_depth + l_rppg

            # ---------------- BACKPROP ---------------- #
            total_loss.backward()
            optimizer.step()

            # ---------------- LOG ---------------- #
            print(f"[{os.path.basename(videoPath)}] Depth: {l_depth.item():.4f} | rPPG: {l_rppg.item():.4f}")

            epoch_loss += total_loss.item()
            count += 1
        vid_count+=1
        if(vid_count%10==0):
                torch.save(cnn_model.state_dict(), f'checkpoint/cnn_epoch_{vid_count}.pth')
                torch.save(rnn_model.state_dict(), f'checkpoint/rnn_epoch_{vid_count}.pth')


    avg_loss = epoch_loss / (count + 1e-6)

    print(f"\nEpoch [{epoch}/{epochs}] | Avg Loss: {avg_loss:.4f}\n")




# ---------------- FINAL SAVE ---------------- #
torch.save(cnn_model.state_dict(), 'final_models/karnot_cnn_final.pth')
torch.save(rnn_model.state_dict(), 'final_models/woffman_lstm_final.pth')