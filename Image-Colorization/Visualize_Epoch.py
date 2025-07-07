import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.color import lab2rgb
import time
import os

def lab_to_rgb(L, ab):
    
    
    # [-1, 1] to [0, 100]
    L = (L + 1) * 50
    #  [-1, 1] to [-128, 127]
    ab = (ab + 1) * 255 / 2 - 128
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img) 
        rgb_imgs.append(img_rgb) 
    return np.stack(rgb_imgs, axis=0)

def visualize(model, data, epoch=None, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L

    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)

    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")

    plt.show()

    save_dir = "./Saved_Images"
    os.makedirs(save_dir, exist_ok=True)

    if save:
        if epoch is not None:
            save_path = os.path.join(save_dir, f"colorization_epoch_{epoch}.png")
        else:
            save_path = os.path.join(save_dir, f"colorization_{int(time.time())}.png")

        fig.savefig(save_path)
        print(f"Görsel kaydedildi: {save_path}")
