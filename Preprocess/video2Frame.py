import numpy as np
import cv2
from collections import deque

def createVideoBatchOfFive(path, size=5, frame_skip=2):

    cap = cv2.VideoCapture(path)
    fw = deque(maxlen=size)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize
        reFrame = cv2.resize(frame, (256, 256))

        # RGB + HSV
        rgb = cv2.cvtColor(reFrame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(reFrame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        combined_6ch = np.concatenate([rgb, hsv], axis=2)

        fw.append((combined_6ch, reFrame))

        if len(fw) == size:
            batch_data = list(fw)

            sequence_tensor = np.stack(
                [item[0] for item in batch_data], axis=0
            )

            sequence_tensor = np.ascontiguousarray(sequence_tensor)

            mid_frame = batch_data[size // 2][1]

            yield sequence_tensor, mid_frame

            fw.clear()  # 🔥 remove overlap (training)

    cap.release()