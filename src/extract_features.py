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
        print("using m1 GPU acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("using nvidia gpu acceleration")
        return torch.device("cuda")
    else:
        print("no gpu detected > using cpu")
        return torch.device("cpu")
    
device = get_device()

#load resnet, remove classification head and return the 2048-dim feature extractor
def get_resnet_feature_extractor():
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights = weights)
    except:
        model = models.resnet50(pretrained=True) #fallback to older versions

    modules = list(model.children())[:-1] #remove the last fc layer
    model = nn.Sequential(*modules) # output = (batch, 2048, 1, 1)

    model.to(device)
    model.eval() #freeze layers
    return model    
