import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from collections import deque

from model.CNN import Karnot
from model.LSTM import WoffMan


# ---------------- MASKING ---------------- #
def non_rigid_registration(feature_map, pred_depth, threshold=0.1):
    mask = (pred_depth > threshold).float()
    return feature_map * mask


# ---------------- DEPTH VISUALIZATION ---------------- #
def visualize_depth(depth_tensor):
    """
    depth_tensor: (1, 32, 32)
    """
    depth = depth_tensor.squeeze().cpu().numpy()

    # Normalize to 0–255
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = (depth * 255).astype(np.uint8)

    # Resize for display
    depth = cv2.resize(depth, (256, 256))

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    return depth_colored


# ---------------- TEST FUNCTION ---------------- #
def test_video(video_path, cnn_path, lstm_path, device):

    # MUST match training
    cnn = Karnot().to(device).float()
    lstm = WoffMan(input_dim=128, fs=35).to(device).float()

    # -------- LOAD MODELS -------- #
    try:
        cnn_ckpt = torch.load(cnn_path, map_location=device)

        if isinstance(cnn_ckpt, dict) and 'model_state_dict' in cnn_ckpt:
            cnn.load_state_dict(cnn_ckpt['model_state_dict'])
        else:
            cnn.load_state_dict(cnn_ckpt)

        lstm.load_state_dict(torch.load(lstm_path, map_location=device))

    except Exception as e:
        print(f"Error loading models: {e}")
        return

    cnn.eval()
    lstm.eval()

    cap = cv2.VideoCapture(video_path)
    fw = deque(maxlen=5)

    final_scores = []

    print(f"Analyzing Video: {video_path}...")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # -------- PREPROCESS -------- #
            reFrame = cv2.resize(frame, (256, 256))

            rgb = cv2.cvtColor(reFrame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(reFrame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

            combined = np.concatenate([rgb, hsv], axis=2).copy()

            fw.append(torch.from_numpy(combined).permute(2, 0, 1))

            if len(fw) == 5:

                input_seq = torch.stack(list(fw)).unsqueeze(0).to(device).float()
                B, T, C, H, W = input_seq.shape

                # -------- CNN -------- #
                pred_depths, raw_features = cnn(input_seq.squeeze(0))
                # pred_depths: (T,1,32,32)
                # raw_features: (T,128,32,32)

                # -------- MASKING -------- #
                registered_features = non_rigid_registration(raw_features, pred_depths)

                # -------- POOLING -------- #
                pooled = F.adaptive_avg_pool2d(
                    registered_features, (1, 1)
                ).view(1, T, -1)  # (1,5,128)

                # -------- LSTM -------- #
                pred_f = lstm(pooled)

                # -------- SCORE -------- #
                mid = T // 2

                depth_score = torch.norm(pred_depths[mid])
                rppg_score = torch.norm(pred_f)

                score = depth_score + 0.015 * rppg_score
                final_scores.append(score.item())

                # -------- SHOW DEPTH MAP -------- #
                depth_img = visualize_depth(pred_depths[mid])

                combined_display = np.hstack([reFrame, depth_img])

                cv2.imshow("RGB Frame | Depth Map", combined_display)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()

    # -------- FINAL RESULT -------- #
    if final_scores:
        avg_score = np.mean(final_scores)

        # ⚠️ tune threshold
        result = "REAL (LIVE)" if avg_score > 2.0 else "SPOOF (ATTACK)"

        print(f"Result: {result} | Score: {avg_score:.4f}")
    else:
        print("No frames processed.")


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VIDEO_PATH = r"D:\Storeage-1\Main\ModuleI\Video\0.mp4"
    CNN_PATH = r"D:\Storeage-1\Main\ModuleI\checkpoint\cnn_epoch_150.pth"
    LSTM_PATH = r"D:\Storeage-1\Main\ModuleI\checkpoint\rnn_epoch_150.pth"

    if os.path.exists(VIDEO_PATH):
        test_video(VIDEO_PATH, CNN_PATH, LSTM_PATH, DEVICE)
    else:
        print(f"Video file not found at {VIDEO_PATH}")