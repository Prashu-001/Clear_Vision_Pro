import numpy as np
import time
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
from PIL import Image

lpips_model = LPIPS(net='alex')

def compute_metrics(original, restored):
    # Ensure input range is [0, 255] for PSNR/SSIM and [0,1] for LPIPS
    original = np.clip(((original + 1) * 127.5), 0, 255).astype(np.uint8)
    restored = np.clip(((restored + 1) * 127.5), 0, 255).astype(np.uint8)

    # Resize if too small for SSIM
    min_size = 7
    if original.shape[0] < min_size or original.shape[1] < min_size:
        original = np.resize(original, (min_size, min_size, 3))
        restored = np.resize(restored, (min_size, min_size, 3))

    psnr_val = psnr(original, restored, data_range=255)
    ssim_val = ssim(original, restored, channel_axis=2, data_range=255)

    # LPIPS expects tensors in [-1,1], shape (1,3,H,W)
    original_lp = torch.tensor((original / 127.5) - 1.0).permute(2, 0, 1).unsqueeze(0).float()
    restored_lp = torch.tensor((restored / 127.5) - 1.0).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        lpips_val = lpips_model(original_lp, restored_lp).item()

    return psnr_val, ssim_val, lpips_val


