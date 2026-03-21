import subprocess, base64, re
import numpy as np
import matplotlib.pyplot as plt

def get_depth(windows_path):
    wsl_path = windows_path.replace('D:', '/mnt/d').replace('\\', '/')
    cmd = ["wsl", "-d", "Ubuntu-22.04", "python3", "/home/anuj/Project/PRNet-Depth-Generation/dmHookP.py", wsl_path]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = proc.communicate()

    match = re.search(r"FACE_DEPTH_B64:(.*)", stdout)
    if match:
        b64_data = match.group(1).strip()
        binary = base64.b64decode(b64_data)
        return np.frombuffer(binary, dtype=np.float32).reshape(32, 32)
    return None

# Execution
img_path = r"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
depth = get_depth(img_path)

if depth is not None:
    # Use nipy_spectral for higher detail visibility
    plt.imshow(depth, cmap='nipy_spectral', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Fixed 32x32 Face Depth")
    plt.show()