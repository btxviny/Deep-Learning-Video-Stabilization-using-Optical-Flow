import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from time import time 
from torchvision import transforms, models

from utils import *
from model import MPI_Net
from feature_stabilization import feature_stabilization

import matplotlib.pyplot as plt
device = 'cuda'
shape = (H,W,C) = (128,128,3)
start = time()




def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using StabNet')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()



def stabilize(in_path,out_path):
    
    if not os.path.exists(in_path):
        print(f"The input file '{in_path}' does not exist.")
        exit()
    _,ext = os.path.splitext(in_path)
    if ext not in ['.mp4','.avi']:
        print(f"The input file '{in_path}' is not a supported video file (only .mp4 and .avi are supported).")
        exit()

    #Load frames and stardardize
    cap = cv2.VideoCapture(in_path)
    frames = []
    while True:
        ret,frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    frames = np.array(frames, np.uint8)
    n_frames,height,width,_ = frames.shape
    print(f'Feature Stabilization\n')
    frames = feature_stabilization(frames,window_size=15)
    print(f'Inpainting Flows\n')
    pca_flows, masks = inpaint_flows(frames, raft, midas, midas_transforms)
    # Initialize final_flows with the original optical flow fields (shape: [n_frames, h, w, 2])
    in_flows = pca_flows.copy() 
    warps = np.zeros_like(pca_flows)
    window_size = 20

    # Loop through the optical flow fields using a sliding window approach
    prev_warp = np.zeros((H,W,2),dtype=np.float32)
    for idx in range(1,n_frames - window_size, 1):
        # Extract the current window of optical flow fields
        net_in = in_flows[idx : idx + window_size]
        net_in[0,...] -= prev_warp
        net_in = np.concatenate(net_in, axis=-1)
        net_in = torch.from_numpy(net_in).permute(2, 0, 1).unsqueeze(0).float().cuda()
        with torch.no_grad():
            net_out = model(net_in)
        dx = net_out[:, ::2, :, :]
        dy = net_out[:, 1::2, :, :]
        w = torch.stack([dx, dy], dim=-1).squeeze(0).to('cpu').numpy()
        prev_warp = w[0,...]
        warps[idx,...] = prev_warp
        flo_rgb = show_flow(w[0,...])
        cv2.imshow('window',flo_rgb)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()

    print(f'\nLow-Pass Filter\n')    
    warps = temp_smooth(pca_flows,warps)
    
    #warp and write to file
    frame_count, h, w, c = frames.shape
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
    for idx in range(frame_count-1):
        frame = torch.from_numpy(frames[idx,...]/255.0).unsqueeze(0).permute(0,3,1,2).float().cuda()
        flow = torch.from_numpy(warps[idx,...]).unsqueeze(0).permute(0,3,1,2).float().cuda()
        flow = gaussian_blur_upscale(flow,kernel_size = 5, sigma = 1, size = (height,width))
        warped_image = dense_warp(frame,flow).cpu().squeeze(0).permute(1,2,0).numpy() 
        warped_image = (warped_image * 255).astype(np.uint8)
        out.write(warped_image)
        cv2.imshow('window',fixBorder(warped_image))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    args = parse_args()
    #in_path = 'E:/Datasets/DeepStab_Dataset/unstable/18.avi'
    #out_path = './2.avi'
    #set up raft
    raft = models.optical_flow.raft_small(weights = 'Raft_Small_Weights.C_T_V2').eval().to(device)
    #set up midas
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type).eval().to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    ckpt_dir = './ckpts/'
    model = MPI_Net(input_channels=40,num_outputs=40, ngf=32).eval().to(device)
    ckpts = os.listdir(ckpt_dir)
    if ckpts:
        ckpts = sorted(ckpts, key = lambda x : int(x.split('.')[0].split('_')[1])) #sort
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(ckpt_dir,latest))
        model.load_state_dict(state_dict['model'])
        print(f'loaded weights {latest}')
    stabilize(args.in_path, args.out_path)