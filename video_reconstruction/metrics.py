import torch
from kornia.losses import ssim as dssim
from skimage.metrics import structural_similarity
from einops import rearrange
import numpy as np

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    return structural_similarity(image_pred.cpu().numpy(), image_gt, win_size=11, multichannel=True, gaussian_weights=True)

def lpips(image_pred, image_gt, lpips_model):
    gt_lpips = image_gt * 2.0 - 1.0
    gt_lpips = rearrange(gt_lpips, '(b h) w c -> b c h w', b=1)
    gt_lpips = torch.from_numpy(gt_lpips)
    predict_image_lpips = image_pred.clone().detach().cpu() * 2.0 - 1.0
    predict_image_lpips = rearrange(predict_image_lpips, '(b h) w c -> b c h w', b=1)
    lpips_result = lpips_model.forward(predict_image_lpips, gt_lpips).cpu().detach().numpy()
    return np.squeeze(lpips_result)
