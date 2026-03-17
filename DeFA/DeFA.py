import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
model.to(device)
model.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform



PATH_VIDEO = r"D:\Storeage-1\Main\ModuleI\Video\know.mp4"

# Open video
cap = cv2.VideoCapture(PATH_VIDEO)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transform
    input_batch = transform(img).to(device)

    # Depth prediction
    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize depth for visualization1
    depth = depth.astype(np.uint8)

    # Apply colormap
    depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    # depth_colormap = (depth_colormap>120).astype(int)
    print( depth_colormap.shape)

    # Show results
    cv2.imshow("Original", frame)
    cv2.imshow("Depth", depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()