import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

def get_depth_from_wsl(windows_path):
    # 1. Path Conversion (D: -> /mnt/d)
    # Ensure your drive is actually /mnt/d by checking 'ls /mnt' in WSL
    wsl_path = windows_path.replace('D:', '/mnt/d').replace('\\', '/')
    
    # 2. Command to run in your specific Ubuntu-22.04 distro
    # We call the hook script using its absolute WSL path
    cmd = ["wsl", "-d", "Ubuntu-22.04", "python3", "/home/anuj/Project/3DDFA_V2/dmHook.py", wsl_path]
    
    print(f"Requesting depth for: {wsl_path}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print WSL logs for debugging (this will show "No face detected" etc.)
    if stderr:
        print(f"--- WSL DEBUG ---\n{stderr.decode()}")

    if not stdout or len(stdout) < 4096:
        print(f"Error: Received invalid data ({len(stdout)} bytes).")
        return None

    try:
        # Reconstruct the 32x32 float32 array
        depth_map = np.frombuffer(stdout[:4096], dtype=np.float32).reshape(32, 32)
        return depth_map
    except Exception as e:
        print(f"Conversion Error: {e}")
        return None

# --- EXECUTE TEST ---
img_path = r"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
depth = get_depth_from_wsl(img_path)
# if os.path.exists(img_path):
    
#     if depth is not None and not np.all(depth == 0):
#         print("Success! RDJ Depth Map received.")
#         plt.imshow(depth, cmap='jet')
#         plt.colorbar(label='Normalized Depth (Z)')
#         plt.title("32x32 Spatial Supervision Map")
#         plt.show()
#     else:
#         print("Hook returned zeros. Check the WSL Debug logs above.")
# else:
#     print(f"File not found on Windows: {img_path}")