import subprocess, base64, re
import numpy as np
import matplotlib.pyplot as plt

def get_deca_depth(windows_path):
    # Convert Windows path to WSL mount path
    wsl_path = windows_path.replace('D:', '/mnt/d').replace('\\', '/')
    
    # Target the new DECA hook
    cmd = ["wsl", "-d", "Ubuntu-22.04", "python3", "/home/anuj/Project/DECA/DmHook.py", wsl_path]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()

    match = re.search(r"FACE_DEPTH_B64:(.*)", stdout)
    if match:
        b64_data = match.group(1).strip()
        binary = base64.b64decode(b64_data)
        return np.frombuffer(binary, dtype=np.float32).reshape(32, 32)
    
    if stderr: print(f"WSL Error: {stderr}")
    return None

# Execution
img_path = r"D:\Storeage-1\Main\ML-Model\examps\RDJ.jpg"
depth = get_deca_depth(img_path)

if depth is not None:
    plt.imshow(depth, cmap='nipy_spectral', vmin=0, vmax=1)
    plt.colorbar(label="Normalized Depth (Pseudo-Z)")
    plt.title("DECA-Generated 32x32 Auxiliary Depth")
    plt.show()