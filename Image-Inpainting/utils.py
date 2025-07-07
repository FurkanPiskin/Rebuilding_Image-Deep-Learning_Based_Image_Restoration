import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import model.data as loader

def normalize_tensor(data: torch.Tensor,
                     smin: float,
                     smax: float,
                     tmin : float,
                     tmax : float) -> torch.Tensor:
   

    tmin, tmax = min(tmin, tmax), max(tmin, tmax)
    smin, smax = min(smin, smax), max(smin, smax)

    source_len = smax - smin
    target_len = tmax - tmin

    data = (data - smin) / source_len
    data = (data * target_len) + tmin

    return data

def load_image(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB'ye çevir
    img = torch.from_numpy(img).float()  # torch tensora çevir
    return img


def load_random_mask(image_shape):
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = loader.generate_random_mask(width=width, height=height)  # PIL mask
    mask = np.array(mask).astype(np.float32)  # numpy'a çevir
    return torch.from_numpy(mask)  # tensora çevir

