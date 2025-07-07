import cv2
import torch
import numpy as np
import pytorch_lightning as pl
import os
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Data paths

TRAIN_DATASET_ROOT = Path('')#Train Dataset path
VALID_DATASET_ROOT = Path('')#Valid Dataset path
TEST_DATASET_ROOT = Path('')#Test Dataset path


for path in [TRAIN_DATASET_ROOT, VALID_DATASET_ROOT, TEST_DATASET_ROOT]:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

def generate_random_mask(height=256, width=256, min_lines=1, max_lines=4, 
                        min_vertex=5, max_vertex=13, mean_angle=2/5*math.pi, 
                        angle_range=2/15*math.pi, min_width=10, max_width=25):
    """Generate random irregular mask"""
    mask = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    num_lines = np.random.randint(min_lines, max_lines)
    avg_radius = math.sqrt(height**2 + width**2) / 8

    for _ in range(num_lines):
        
        vertices = []
        start_x, start_y = np.random.randint(0, width), np.random.randint(0, height)
        vertices.append((start_x, start_y))
        
        for _ in range(np.random.randint(min_vertex, max_vertex)):
            angle = np.random.uniform(mean_angle - angle_range, mean_angle + angle_range)
            radius = np.clip(np.random.normal(avg_radius, avg_radius/2), 0, 2*avg_radius)
            new_x = np.clip(vertices[-1][0] + radius * math.cos(angle), 0, width)
            new_y = np.clip(vertices[-1][1] + radius * math.sin(angle), 0, height)
            vertices.append((int(new_x), int(new_y)))

        # Draw line
        line_width = int(np.random.uniform(min_width, max_width))
        draw.line(vertices, fill=1, width=line_width)
        
        # Smooth corners
        for (x, y) in vertices:
            draw.ellipse([x-line_width//2, y-line_width//2, 
                         x+line_width//2, y+line_width//2], fill=1)

    # Random flips
    if np.random.random() > 0.5:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.random() > 0.5:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    return torch.from_numpy(np.array(mask, np.float32))






def generate_combined_mask(image_width=256, image_height=256, random_prob=0.5, flip_prob=0.5):
    """
    Yüz datası için kullanılmıştır.
    Eğitim sırasında hem göz maskesi hem de rastgele maskeleri karıştırarak üretir.
    - random_prob: rastgele maskeye geçiş ihtimali (örneğin %50)
    - flip_prob: maskeye yatay flip uygulama olasılığı
    """

    def generate_eye_mask(eye='right'):
        top = int(image_height * 0.35)
        bottom = int(image_height * 0.50)
        eye_width = int(image_width * 0.18)
        eye_height = bottom - top

        if eye == 'right':
            left = int(image_width * 0.60)
        else:
            left = int(image_width * 0.22)

        right = left + eye_width

        mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, right, bottom], fill=255)
        return mask

    def generate_random_line_mask():
        mask = Image.new('1', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)

        num_lines = np.random.randint(1, 4)
        base_radius = math.sqrt(image_height**2 + image_width**2) / 8
        avg_radius = base_radius * 0.6

        for _ in range(num_lines):
            vertices = []
            x0, y0 = np.random.randint(0, image_width), np.random.randint(0, image_height)
            vertices.append((x0, y0))

            for _ in range(np.random.randint(5, 13)):
                angle = np.random.uniform(2/5*math.pi - 2/15*math.pi,
                                          2/5*math.pi + 2/15*math.pi)
                radius = np.clip(np.random.normal(avg_radius, avg_radius / 2), 0, 2 * avg_radius)
                new_x = np.clip(vertices[-1][0] + radius * math.cos(angle), 0, image_width)
                new_y = np.clip(vertices[-1][1] + radius * math.sin(angle), 0, image_height)
                vertices.append((int(new_x), int(new_y)))

            line_w = max(1, int(np.random.uniform(4, 12)))
            draw.line(vertices, fill=1, width=line_w)

            for (x, y) in vertices:
                draw.ellipse(
                    [x - line_w // 2, y - line_w // 2,
                     x + line_w // 2, y + line_w // 2],
                    fill=1
                )

        return mask.convert('L')

    # Rastgele maske türünü seç (göz maskesi veya rastgele çizgi)
    if random.random() < random_prob:
        mask_img = generate_random_line_mask()
    else:
        mask_img = generate_eye_mask(eye=random.choice(['left', 'right']))

    # Yatay flip uygulanacak mı?
    if random.random() < flip_prob:
        mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)

    # Maske tensor olarak dönsün (float32, 0–1 aralığında)
    mask_np = (np.array(mask_img) > 0).astype(np.float32)
    return torch.from_numpy(mask_np)


class MaskedDataset(Dataset):
    def __init__(self, root_dir, img_size=256, transform=None):
        self.root = Path(root_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            self.image_files.extend(self.root.glob(ext))
            
        if not self.image_files:
            raise ValueError(f"No images found in {root_dir}")
            
        print(f"Loaded {len(self.image_files)} images from {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = str(self.image_files[idx])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read {img_path}")
            
        # Convert and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC --> CHW
        #image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
        
        # Generate mask
        mask = generate_random_mask()
        #mask=generate_combined_mask()
        
        if self.transform:
            image = self.transform(image)
            
        return image.float(), mask.float()

class PlacesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, num_workers=4, img_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

    def setup(self, stage=None):
        self.train_ds = MaskedDataset(TRAIN_DATASET_ROOT, self.img_size)
        self.val_ds = MaskedDataset(VALID_DATASET_ROOT, self.img_size)
        self.test_ds = MaskedDataset(TEST_DATASET_ROOT, self.img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

def show_masked_image(image, mask):
    
    # Tensor --> NumpyArray
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))  # CHW-->HWC
        image = (image + 1) / 2  # [-1,1] to [0,1]
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Maskeyi uygula
    masked = image * (1 - mask[..., np.newaxis])
    
    # Görseleştir
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.clip(0, 1))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(mask.squeeze(), cmap='gray')
    ax[1].set_title("Mask")
    ax[1].axis('off')
    
    ax[2].imshow(masked.clip(0, 1))
    ax[2].set_title("Masked Image")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
  
    dataset = MaskedDataset(TRAIN_DATASET_ROOT)
    #print(f"Dataset length: {len(dataset)}")
    
   
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
  
    images, masks = next(iter(dataloader))
    #print(f"Batch shape - images: {images.shape}, masks: {masks.shape}")
    
    
    show_masked_image(images[0], masks[0])