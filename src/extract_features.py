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

#reads video, extracts 1fps, runs resnet and returens numpy array of shape (n_seconds, 2048)
def extract_features_for_video(video_path, model, transform):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"error opening video {video_path}")
        return None
    
    fps = capture.get(cv2.CAP_PROP_FPS) #calculate frame interval for 1fps
    if fps <= 0:
        fps = 25.0 #fallback

    frame_interval = int(np.round(fps)) #skip fps frames
    frames_buffer = []
    features_list = []
    frame_count = 0

    while True:
        retain, frame = capture.read()
        if not retain:
            break

        #downsampling to 1 fps
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert bgr to rgb
            
            frame_tensor = transform(frame) #preprocess to tensor
            frames_buffer.append(frame_tensor)

            if len(frames_buffer) == batch_size: #when batch is full
                batch = torch.stackl(frames_buffer).to(device) #stack (32, 3, 224, 224)

                with torch.no_grad():
                    output = model(batch) #(32, 2048, 1, 1)
                    output = output.flatten(start_dim=1) #(32, 2048)

                features_list.append(output.cpu().numpy())
                frames_buffer = []
        
        frame_count += 1

    if frames_buffer: #remaining frames
        batch = torch.stack(frames_buffer).to(device)
        with torch.no_grad():
            output = model(batch).flatten(start_dim=1)
        features_list.append(output.cpu().numpy())

    capture.release()

    if not features_list:
        return None
    
    return np.concatenate(features_list, axis=0) #concatenate all batches into 1 long array
        

