import cv2
import torch
import numpy as np
from 3DDFA_V2 import TDDFA
from utils.depth import depth as render_depth

# 1. Initialize the DeFA (3DDFA) model
config = 'configs/mb1_120x120.yml' # MobileNet backbone for speed
tddfa = TDDFA(gpu_mode=True, config=config)

def generate_defa_depth(image_path):
    img = cv2.imread(image_path)

    boxes = face_boxes(img) 
    param_lst, roi_box_lst = tddfa(img, boxes)
    
    
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    
    raw_depth = render_depth(img, ver_lst, tddfa.tri, show_flag=False)
    
    normalized_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-7)
    final_depth = cv2.resize(normalized_depth, (32, 32))
    
    return final_depth