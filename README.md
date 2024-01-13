# Deep-Learning-Video-Stabilization-using-Optical-Flow

This is a PyTorch implementation of the paper [Learning Video Stabilization Using OpticalFlow](https://cseweb.ucsd.edu/~ravir/jiyang_cvpr20.pdf).

This stabilization algorithm is based on pixel-profile stabilization. The pixel profiles are constructed using the estimated dense optical flow. Before using the pretrained network to stabilize this profiles we must exclude dynamically moving objects by inpainting their optical flow with continuous values.
