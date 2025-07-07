import os
import glob
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab,lab2rgb
import torch
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import warnings

class ColorizationDataset(Dataset):
    
 

    def __init__(self, paths, crop_size):
       
        self.paths = paths
        self.size = crop_size

    def __getitem__(self, idx):
        
          # Görseli aç ve RGB formatına dönüştür
        img = Image.open(self.paths[idx]).convert("RGB")
        
        # Görseli belirtilen boyuta getir (isteğe bağlı, şu an devre dışı)
        #img = transforms.Compose([transforms.CenterCrop(self.size)])(img)
        
        # Görseli NumPy dizisine çevir
        img = np.array(img)
        
         # RGB görseli L*a*b renk uzayına dönüştür
        img_lab = rgb2lab(img).astype("float32")
        
          # Görseli PyTorch tensörüne çevir
        img_lab = transforms.ToTensor()(img_lab)
        
        # L kanalını (aydınlık bilgisi) -1 ile 1 arasına normalize et
        # Orijinal aralık: [0, 100], Hedef aralık: [-1, 1]
        L = img_lab[0, ...] / 50. - 1.  
          # çıktı boyutu: (batch_size, H, W)
        
         # a ve b kanallarını (renk bilgisi) -1 ile 1 arasına normalize et
        # Orijinal aralık: [-128, 127], Hedef aralık: [-1, 1]
        ab = (img_lab[1:3, ...] + 128.) / 255. * 2. - 1.
        # çıktı boyutu: (batch_size, channels, H, W)

        
        return {'L': L.unsqueeze(0), 'ab': ab}

    def __len__(self):
       
        return len(self.paths)