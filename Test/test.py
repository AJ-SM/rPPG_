import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Import your model structures
from model.CNN import Karnot
from model.LSTM import WoffMan

def test_video(video_path, cnn_path, lstm_path, device):
    # 1. Initialize and Load Models
    cnn = Karnot().to(device).float()
    lstm = WoffMan(input_dim=1).to(device).float()
    
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    
    cnn.eval()
    lstm.eval()

    cap = cv2.VideoCapture(video_path)
    fw = deque(maxlen=5)
    
    scores = []
    depth_maps = []

    print(f"Analyzing Video: {video_path}...")

    with torch.no_grad(): # Disable gradient calculation for speed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Pre-process exactly like training
            reFrame = cv2.resize(frame, (256, 256))
            fw.append(reFrame)

            if len(fw) == 5:
                batch = list(fw)
                mid_frm = batch[2]

                # Create RGB + HSV 6-channel input
                mid_rgb = cv2.cvtColor(mid_frm, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                mid_hsv = cv2.cvtColor(mid_frm, cv2.COLOR_BGR2HSV).astype(np.float32)/255.0
                input_tensor = np.concatenate([mid_rgb, mid_hsv], axis=2)

                # Convert to Torch Tensor (1, 6, 256, 256)
                input_pt = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0).to(device).float()

                # Forward Pass
                pred_depth, raw_features = cnn(input_pt)
                pooled = F.adaptive_avg_pool2d(raw_features, (1, 1)).view(1, 1, -1)
                pred_pulse = lstm(pooled)

                # Collect results
                # In the paper, a 'Live' face has a pulse variance and 3D shape
                # Here we take the absolute value of the pulse prediction as a confidence score
                print(pred_pulse.item())
                scores.append(pred_pulse.item())
                
                if len(depth_maps) < 1: # Save one depth map for visual check
                    depth_maps.append(pred_depth[0, 0].cpu().numpy())

    cap.release()

    # --- FINAL DECISION LOGIC ---
    # We average the temporal pulse scores. 
    # Real faces typically produce consistent, non-zero pulse waves.
    avg_score = np.mean(np.abs(scores))
    threshold = 0.7 # You can tune this based on your validation results
    
    label = "REAL (LIVE)" if avg_score > threshold else "SPOOF (ATTACK)"
    print(f"\nResult: {label}")
    print(f"Confidence Score: {avg_score:.4f}")

    # Visual Verification
    if depth_maps:
        plt.subplot(1, 2, 1)
        plt.title("Input Frame")
        plt.imshow(cv2.cvtColor(reFrame, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 2, 2)
        plt.title("Predicted Depth")
        plt.imshow(depth_maps[0], cmap='jet')
        plt.show()

# --- RUN TEST ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_TO_TEST = r"D:\Storeage-1\Main\ModuleI\Video\rko.mp4"
CNN_WEIGHTS = r"D:\Storeage-1\Main\ModuleI\final_models\karnot_cnn_final.pth"
LSTM_WEIGHTS = r"D:\Storeage-1\Main\ModuleI\final_models\woffman_lstm_final.pth"

test_video(VIDEO_TO_TEST, CNN_WEIGHTS, LSTM_WEIGHTS, DEVICE)