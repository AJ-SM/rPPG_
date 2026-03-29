import subprocess
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def get_depth_from_wsl(windows_path):
    wsl_path = windows_path.replace('D:', '/mnt/d').replace('\\', '/')
    project_dir = "/home/anuj/Project/3DDFA_V2"
    wsl_cmd = f"cd {project_dir} && python3 dmHook.py {wsl_path}"
    cmd = ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", wsl_cmd]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if len(stdout) < 4096: return None

    try:
        depth_map = np.frombuffer(stdout[:4096], dtype=np.float32).reshape(32, 32)
        return depth_map
    except: return None

# --- RUN VERIFICATION ---
# img_path = r"D:\Storeage-1\Main\ModuleI\DeFA\images\1.PNG"

# if os.path.exists(img_path):
#     depth = get_depth_from_wsl(img_path)
    
#     if depth is not None:
#         # Load and resize original to 32x32
#         img = cv2.imread(img_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_32 = cv2.resize(img_rgb, (32, 32))

#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
#         axes[0].imshow(img_32)
#         axes[0].set_title("Original (32x32 Resized)")
        
#         # Overlay a grid to check alignment
#         axes[1].imshow(depth, cmap='jet')
#         axes[1].set_title("Updated Depth Map (Scaled to Image)")
        
#         plt.suptitle("Check if the Face Size Matches the Markings")
#         plt.show()






























# import subprocess
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def get_depth_from_wsl(windows_path):
#     # 1. Path Conversion: Match your /mnt/d mount
#     wsl_path = windows_path.replace('D:', '/mnt/d').replace('\\', '/')
    
#     # 2. Construct the Bash Bridge command
#     # We 'cd' into the directory first to prevent ModuleNotFoundErrors
#     project_dir = "/home/anuj/Project/3DDFA_V2"
#     wsl_cmd = f"cd {project_dir} && python3 dmHook.py {wsl_path}"
    
#     # 3. Call specific distribution
#     cmd = ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", wsl_cmd]
    
#     # print(f"Executing: {' '.join(cmd)}")
    
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()

#     # Print WSL logs (stderr) to your Windows console
#     if stderr:
#         print(f"--- WSL DEBUG LOGS ---\n{stderr.decode()}")

#     # Check if we got the expected 4096 bytes (32*32*4)
#     if len(stdout) < 4096:
#         print(f"Error: Incomplete data. Received {len(stdout)} bytes.")
#         return None

#     try:
#         # Reconstruct the array
#         depth_map = np.frombuffer(stdout[:4096], dtype=np.float32).reshape(32, 32)
#         return depth_map
#     except Exception as e:
#         print(f"Reconstruction Error: {e}")
#         return None

# # --- TEST ---
# img_path = r"D:\Storeage-1\Main\ModuleI\DeFA\images\2323.PNG"

# if os.path.exists(img_path):
#     depth = get_depth_from_wsl(img_path)
    
#     if depth is not None:
#         print("Success! Displaying Depth Map...")
#         plt.imshow(depth, cmap='jet')
#         plt.title("32x32 Face Depth (Live)")
#         plt.colorbar()
#         plt.show()
# else:
#     print(f"Windows cannot find file: {img_path}")