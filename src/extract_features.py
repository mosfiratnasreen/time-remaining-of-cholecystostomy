import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2  # reads videos
import numpy as np
import os
from tqdm import tqdm  # progress bar

video_dir = "data/cholec80/videos"
output_dir = "data/features"
batch_size = 32

def get_device():
    if torch.backends.mps.is_available():
        print("uising m1 GPU acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("using nvidia gpu acceleration")
        return torch.device("cuda")
    else:
        print("no gpu detected > using cpu")
        return torch.device("cpu")
    
