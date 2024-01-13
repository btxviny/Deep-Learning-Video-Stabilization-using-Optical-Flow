# Deep-Learning-Video-Stabilization-using-Optical-Flow

This is a PyTorch implementation of the paper [Learning Video Stabilization Using OpticalFlow](https://cseweb.ucsd.edu/~ravir/jiyang_cvpr20.pdf).

This stabilization algorithm is based on pixel-profile stabilization. The pixel profiles are constructed using the estimated dense optical flow. Before using the pretrained network to stabilize these profiles we must exclude dynamically moving objects by inpainting their optical flow with continuous values. We identify sych objects through a cascade of conditions as described in the paper with the modification that I include a depth mask as shown in the figure below.
![mask](https://github.com/btxviny/Deep-Learning-Video-Stabilization-using-Optical-Flow/blob/main/images/mask.png)
