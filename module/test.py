import cv2 as cv
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ImgtorPPG import   process_rppg_pipeline   # your file name


VIDEO_PATH = r"D:\Storeage-1\Main\ModuleI\Video\baddiprinted.mp4"



def extract_data_from_video(video_path):
    model_path =  r"D:\Storeage-1\Main\ModuleI\reqModels\face_landmarker.task"

    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS)) or 30

    frames = []
    masks = []
    embeddings = []

    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    timestamp_ms = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        h, w, _ = frame.shape

        mask = np.zeros((h, w), dtype=np.uint8)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(mp_image, timestamp_ms)

        timestamp_ms += int(1000 / fps)

        embedding = None

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            points = []

            for lm in face_landmarks:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            cv.rectangle(
                mask,
                (min(xs), min(ys)),
                (max(xs), max(ys)),
                255,
                -1
            )

            embedding = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks]
            ).flatten()

        frames.append(frame)
        masks.append(mask)
        embeddings.append(embedding)

    cap.release()

    return frames, masks, embeddings, fps


frames, masks, embeddings, fps = extract_data_from_video(
    VIDEO_PATH,
    
)

output = process_rppg_pipeline(
    frames,
    masks,
    embeddings,
    fps
)

print("Heart Rate Results")
print("-------------------")

for algo in output["results"]:
    hr = output["results"][algo]["heart_rate"]
    print(algo, ":", round(hr, 2), "BPM")