import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

def get_depth_from_wsl(windows_path):
    # 1. Path Conversion: Match your /mnt/d mount
    wsl_path = windows_path.replace('D:', '/mnt/d').replace('\\', '/')
    
    # 2. Construct the Bash Bridge command
    # We 'cd' into the directory first to prevent ModuleNotFoundErrors
    project_dir = "/home/anuj/Project/3DDFA_V2"
    wsl_cmd = f"cd {project_dir} && python3 dmHook.py {wsl_path}"
    
    # 3. Call specific distribution
    cmd = ["wsl", "-d", "Ubuntu-22.04", "bash", "-c", wsl_cmd]
    
    # print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print WSL logs (stderr) to your Windows console
    if stderr:
        print(f"--- WSL DEBUG LOGS ---\n{stderr.decode()}")

    # Check if we got the expected 4096 bytes (32*32*4)
    if len(stdout) < 4096:
        print(f"Error: Incomplete data. Received {len(stdout)} bytes.")
        return None

    try:
        # Reconstruct the array
        depth_map = np.frombuffer(stdout[:4096], dtype=np.float32).reshape(32, 32)
        return depth_map
    except Exception as e:
        print(f"Reconstruction Error: {e}")
        return None

# # --- TEST ---
# img_path = r"D:\Storeage-1\Main\ModuleI\dataSet\raw\ImposterRaw\0006\0006_00_00_01_30.jpg"

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