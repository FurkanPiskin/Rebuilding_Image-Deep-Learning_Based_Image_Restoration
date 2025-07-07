import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from PIL import Image
import sys
from Poisson_Blending import poisson_blend
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import model.gan as gan
import model.data as loader


def load_image(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB'ye çevir
    img = torch.from_numpy(img).float()  # torch tensora çevir
    return img


def calculate_ssim(img1, img2):
    
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    else:
        img1 = img1.astype(np.uint8)

    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)
    else:
        img2 = img2.astype(np.uint8)

   
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = 7 if min_dim >= 7 else min_dim if min_dim % 2 == 1 else min_dim - 1

    ssim_value = ssim(img1, img2, channel_axis=-1, win_size=win_size)
    return ssim_value

def calculate_psnr(img1, img2):
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    else:
        img1 = img1.astype(np.uint8)

    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)
    else:
        img2 = img2.astype(np.uint8)

    psnr_value = psnr(img1, img2)
    return psnr_value


def load_fixed_random_mask(image_shape, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = loader.generate_random_mask(width=width, height=height)  # PIL mask
    mask = np.array(mask).astype(np.float32)
    return torch.from_numpy(mask)

def load_random_mask(image_shape):
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = loader.generate_random_mask(width=width, height=height)  # PIL mask
    mask = np.array(mask).astype(np.float32)  # numpy'a çevir
    return torch.from_numpy(mask)  # tensora çevir


def run(checkpoint_path: str, image_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model yükle
    model = gan.SNPatchGAN.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
  
    # Görsel yükle
    origin_image = load_image(image_path).to(device)
    origin_image = origin_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    
    image_shape = origin_image.shape[2:]  # (H, W)
    
    origin_mask = load_fixed_random_mask(image_shape, seed=42).to(device).unsqueeze(0).unsqueeze(0)

    # GAN çalıştır
    with torch.no_grad():
        print("Input image shape:", origin_image.shape)
        print("Input mask shape:", origin_mask.shape)
        gan_output, coarse_output, refine_output = model(origin_image, origin_mask)
        
       
        gan_output = gan_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        coarse_output = coarse_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        refine_output = refine_output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Görselleri numpy'a çevir ve normalize et
    origin_image_np = origin_image.squeeze(0).permute(1, 2, 0).cpu().numpy() / 255.0
    origin_mask_np = origin_mask.squeeze(0).squeeze(0).cpu().numpy()
    masked_image_np = ((1 - origin_mask_np[..., None]) * origin_image_np)

    # Fonksiyonu uygula
    result = poisson_blend(gan_output, origin_mask_np)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    #Ssim ve Pnsr 
    origin_img_for_metric = origin_image_np * 255.0
    gan_output_for_metric = gan_output

    ssim_val = calculate_ssim(origin_img_for_metric, gan_output_for_metric)
    psnr_val = calculate_psnr(origin_img_for_metric, gan_output_for_metric)

    print(f"SSIM: {ssim_val:.4f}")
    print(f"PSNR: {psnr_val:.2f} dB")


    cv2.imwrite("final_result.jpg", result)
    cv2.imwrite("gan_output.jpg", gan_output)

    # Görselleştir
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(origin_image_np)
    ax[0].set_title('Original image')
    ax[1].imshow(origin_mask_np, cmap='gray')
    ax[1].set_title('Mask')
    ax[2].imshow(masked_image_np)
    ax[2].set_title('Masked image')
    ax[3].imshow(gan_output / 255.0)
    ax[3].set_title('GAN output')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
   
    image_path = ''  # Görsel yolu
    
    checkpoint_path = ''  # Checkpoint yolu

    run(checkpoint_path=checkpoint_path, image_path=image_path)