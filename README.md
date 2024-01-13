# Deep-Learning-Video-Stabilization-using-Optical-Flow

This is a PyTorch implementation of the paper [Learning Video Stabilization Using OpticalFlow](https://cseweb.ucsd.edu/~ravir/jiyang_cvpr20.pdf).

This stabilization algorithm is based on pixel-profile stabilization. The pixel profiles are constructed using the estimated dense optical flow. Before using the pretrained network to stabilize these profiles we must exclude dynamically moving objects by inpainting their optical flow with continuous values. We identify sych objects through a cascade of conditions as described in the paper with the modification that I include a depth mask ,using the midas network, as shown in the figure below.

![mask](https://github.com/btxviny/Deep-Learning-Video-Stabilization-using-Optical-Flow/blob/main/images/mask.png)

The discontinuous regions are inpainted using the principal components of optical flow provided in [PCA-Flow](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf).

![pca](https://github.com/btxviny/Deep-Learning-Video-Stabilization-using-Optical-Flow/blob/main/images/principal%20components.png)

From the inpainted flow we construct the pixel profiles, which are stabilized with our network using a sliding window approach. The network was trained to minimize the following objective function.

![loss](https://github.com/btxviny/Deep-Learning-Video-Stabilization-using-Optical-Flow/blob/main/images/stability_loss.png)

I add a final stabilization step with parametric filtering, using Pytorch's averagepool1d to speed up the process.

1. **Download the pretrained model [Weights](https://drive.google.com/drive/folders/1DNBNRq-ht1NgmPcmOGdwIwjhZs19koic?usp=sharing) and place them in 'ckpts'.**

2. **Run the Stabilization Script:**
   - Run the following command:
     ```bash
     python stabilize.py --in_path unstable_video_path --out_path result_path
     ```
   - Replace `unstable_video_path` with the path to your input unstable video.
   - Replace `result_path` with the desired path for the stabilized output video.

3. I provide the notebook train.ipynb which I used to train the network, with some additional regulatory terms to the loss function so that the resulting video is not oversmoothed.
