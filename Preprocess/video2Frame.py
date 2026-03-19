
import numpy as np
import cv2 
from collections import deque 


def createVideoBatchOfFive(path,size=5):
    cap = cv2.VideoCapture(path)
    fw = deque(maxlen=size)
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret :
            break 
        reFrame = cv2.resize(frame,(256,256))
        fw.append(reFrame)

        if len(fw)==size:
            batch = list(fw)

            mid_frm = batch[2]

            mid_rgb = cv2.cvtColor(mid_frm,cv2.COLOR_BGR2RGB)
            mid_rgb_nrm = mid_rgb.astype(np.float32)/255.0

            mid_hsv = cv2.cvtColor(mid_frm,cv2.COLOR_BGR2HSV)
            mid_hsv_nrm = mid_hsv.astype(np.float32)/255.0

            input_tensor = np.concatenate([mid_rgb_nrm,mid_hsv_nrm],axis=2)

            yield input_tensor,mid_frm
    cap.release()


# --- HOW TO USE IT ---
# video_file = r"D:\Storeage-1\Main\ModuleI\Video\real_client001_android_SD_scene01.mp4"

# for tensor_6ch in createVideoBatchOfFive(video_file):
#     # tensor_6ch is your (256, 256, 6) input
#     # Now you can send 'mid_frame' to your WSL Hook to get the (32, 32) label!
#     print(f"Generated input tensor shape: {len(tensor_6ch)}")
    



