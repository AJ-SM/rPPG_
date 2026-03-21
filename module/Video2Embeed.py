import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_frames_embeddings(video_path, model_path):

    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS)) or 10

    frames = []
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

            embedding = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks],
                dtype=np.float32
            ).flatten()

        frames.append(frame)
        embeddings.append(embedding)

    cap.release()

    return frames, embeddings



# extract_frames_embeddings(r"D:\Storeage-1\Main\ModuleI\Video\nbaddi.mp4",r"D:\Storeage-1\Main\ModuleI\reqModels\face_landmarker.task")