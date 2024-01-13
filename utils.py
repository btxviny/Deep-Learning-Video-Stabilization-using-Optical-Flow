import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class PCA:
    def __init__(self, shape, num_components=5):
        self.shape = shape
        self.num_components = num_components
        U = np.load('./principal_components/PC_U.npy')
        V = np.load('./principal_components/PC_V.npy')
        h, w = shape
        U1 = np.zeros((num_components, h * w))
        V1 = np.zeros((num_components, h * w))
        for i in range(0, num_components):
            temp = U[i, ...].reshape((256, 512))
            temp = cv2.resize(temp, (w, h))
            U1[i, ...] = temp.reshape(1, h * w)
            temp = V[i, ...].reshape((256, 512))
            temp = cv2.resize(temp, (w, h))
            V1[i, ...] = temp.reshape(1, h * w)
        U = torch.from_numpy(U1).float()
        V = torch.from_numpy(V1).float()
        base = torch.cat((U, V), 0).t()
        self.base = base
        self.base.to('cuda')

    def inpaint(self, mask, flow, device='cpu'):
        flow = flow.permute(0, 2, 3, 1).to(device)
        h, w = self.shape
        mask_flat = mask.view(-1).to(device)
        Q = self.base[mask_flat > 0].to(device)
        flow_flat = flow.view(-1, 2).to(device)
        valid_flow = flow_flat[mask_flat > 0]
        c = torch.matmul(torch.matmul((torch.matmul(Q.t(), Q) + 0.1 * torch.eye(10)).inverse(), Q.t()), valid_flow)
        pca_flow = torch.matmul(self.base, c)
        pca_flow = pca_flow.view(h, w, 2).unsqueeze(0)
        mask_tiled = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 2).to(device)
        flow[mask_tiled == 0] = pca_flow[mask_tiled == 0].to(device)
        return flow

def dense_warp(image, flow):
    """
    Densely warps an image using optical flow.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
        flow (torch.Tensor): Optical flow tensor of shape (batch_size, 2, height, width).

    Returns:
        torch.Tensor: Warped image tensor of shape (batch_size, channels, height, width).
    """
    batch_size, channels, height, width = image.size()

    # Generate a grid of pixel coordinates based on the optical flow
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack((grid_x, grid_y), dim=-1).to(image.device)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    new_grid = grid + flow.permute(0, 2, 3, 1)

    # Normalize the grid coordinates between -1 and 1
    new_grid /= torch.tensor([width - 1, height - 1], dtype=torch.float32, device=image.device)
    new_grid = new_grid * 2 - 1
    # Perform the dense warp using grid_sample
    warped_image = F.grid_sample(image, new_grid, align_corners=False)

    return warped_image


def show_flow(flow):
    hsv_mask = np.zeros(shape= flow.shape[:-1] +(3,),dtype = np.uint8)
    hsv_mask[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=True)
    hsv_mask[:,:,0] = ang /2 
    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2RGB)
    return(rgb)

def get_flow(img1, img2, raft):
    img1_t = torch.from_numpy(((img1/255.0) * 2) - 1).permute(-1,0,1).unsqueeze(0).float().to('cuda')
    img2_t = torch.from_numpy(((img2/255.0) * 2) - 1).permute(-1,0,1).unsqueeze(0).float().to('cuda')
    with torch.no_grad():
        flow = raft(img1_t,img2_t)[-1]
    return flow

def fixBorder(frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1 )
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame


def midas_mask(img,midas,transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to('cuda')

    with torch.no_grad():
        depth_map = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = depth_map.float()
    inv_depth = (depth_map.max() - depth_map) / depth_map.max()
    mask = torch.ones_like(inv_depth)
    mask[inv_depth < 0.3] = 0
    mask = mask.int()
    return(mask)

def get_final_mask(mask, img, flo, device = 'cuda'):
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    b, c, h, w = img.shape

    def movingstd2(A, k):
        A = A - torch.mean(A)
        Astd = torch.std(A)
        A = A / Astd
        A2 = A * A
        wuns = torch.ones(A.shape).to(device)
        kernel = torch.ones(2 * k + 1, 2 * k + 1).to(device)
        N = F.conv2d(wuns.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5)
        s = torch.sqrt((torch.nn.functional.conv2d(A2.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5) - ((torch.nn.functional.conv2d(A.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5)) ** 2) / N) / (N - 1))
        s = s * Astd
        return s

    def grad_image(x):
        A = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(device)
        A = A.view((1, 1, 3, 3))
        G_x = F.conv2d(x, A, padding=1)
        B = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(device)
        B = B.view((1, 1, 3, 3))
        G_y = F.conv2d(x, B, padding=1)
        G = torch.sqrt(torch.pow(G_x[0, 0, :, :], 2) + torch.pow(G_y[0, 0, :, :], 2))
        return G

    mean_flo = torch.mean(flo, dim=(2, 3), keepdim=True).to(device)
    mean_flo = torch.ones_like(flo, device=device) * mean_flo
    euclidean_distance = torch.sqrt(torch.sum(torch.pow(flo - mean_flo, 2), dim=1, keepdim=True)).view(h, w)
    u2 = flo[0, 0, :, :]
    v2 = flo[0, 1, :, :]
    A1 = movingstd2(u2, 5).squeeze()
    A2 = movingstd2(v2, 5).squeeze()
    img_gray = torch.mean(img, dim=1, keepdim=True)
    grad = grad_image(img_gray)
    mask[euclidean_distance > 50] = 0
    mask[grad < 0.28] = 0
    mask[A1 > 3 * torch.mean(A1)] = 0
    mask[A2 > 3 * torch.mean(A2)] = 0
    return mask

def inpaint_flows(frames, raft, midas, midas_transforms):
    '''Input:
            frames: [num_frames,height,width,3] numpy
        Output:
            pca_flows [num_frames,128,128,2] numpy
            masks [num_frames,128,128]
    '''
    num_frames = frames.shape[0]
    pca = PCA(shape = (128,128))
    pca_flows = np.zeros((num_frames,128,128,2),dtype = np.float32)
    masks = np.zeros((num_frames,128,128),dtype = np.float32)
    prev = cv2.resize(frames[0,...],(128,128))
    for idx in range(1, num_frames-1):
        curr = cv2.resize(frames[idx,...],(128,128))
        flow = get_flow(prev,curr,raft)
        d_mask = midas_mask(prev, midas, midas_transforms.small_transform)
        mask = get_final_mask(d_mask,prev,flow)
        inpainted_flow = pca.inpaint(mask,flow).cpu().squeeze(0).numpy()
        pca_flows[idx,...] = inpainted_flow
        masks[idx,...] = mask.cpu().numpy()
        prev= curr
        print(f'\r idx: {idx}/{num_frames}',end='')
    return pca_flows, masks

def temp_smooth(pca_flows,warps):
        kernel_size = 15 # must be odd
        device = 'cuda'
        num_frames,H,W,_ = pca_flows.shape
        pca_flows_tensor = torch.from_numpy(pca_flows).float().permute(0, 3, 1, 2).float().to(device)
        warps_tensor = torch.from_numpy(warps).float().permute(0, 3, 1, 2).float().to(device)
        pixel_profiles = torch.cumsum(pca_flows_tensor,dim = 0) 
        smooth_profiles = pixel_profiles.clone() + warps_tensor
        for _ in range(20):
            
            profiles_reshaped =smooth_profiles.permute(2,3,1,0).contiguous().view(-1, 2, num_frames)
            smooth_profiles = torch.nn.functional.avg_pool1d(profiles_reshaped,\
                                                kernel_size = kernel_size,\
                                                stride = 1,
                                                padding = kernel_size //2)
            smooth_profiles = smooth_profiles.permute(-1,1,0).view(num_frames,2,H,W)

        return (smooth_profiles - pixel_profiles).cpu().permute(0,2,3,1).numpy()

def inpaint_results(masks,warps):
    num_frames = masks.shape[0]
    pca = PCA(shape = (128,128))
    #inpaint warp fields
    for idx in range(1,num_frames-1):
        mask = torch.from_numpy(masks[idx,...]).float()
        warp = torch.from_numpy(warps[idx,...]).float().unsqueeze(0).permute(0,3,1,2)
        inpainted_warp = pca.inpaint(mask,warp).cpu().squeeze(0).numpy()
        warps[idx,...] = inpainted_warp
    return warps

def gaussian_blur_upscale(input, kernel_size, sigma, size):
    device  = input.device
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / torch.sum(kernel)

    # Assuming the input is in the shape (batch_size, channels, height, width)
    channels = input.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(1, channels, 1, 1).to(device)

    padding = kernel_size // 2
    blurred = F.conv2d(input, kernel, padding=padding)
    blurred = F.interpolate(blurred,size = size)
    return blurred

