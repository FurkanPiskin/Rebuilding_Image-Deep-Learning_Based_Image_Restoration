import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.color import rgb2lab, lab2rgb
import os
from Unet_Generator import UNetGenerator
from Visualize_Epoch import lab_to_rgb

# Sabit boyut (modelin eğitildiği boyut)
SIZE = 256

# Test klasörü
test_dir = r''

# Yalnızca geçerli resim dosyalarını filtrele
val_paths = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith(('.jpg', '.png'))
]

# Eğer klasörde hiç görsel yoksa hata ver
if not val_paths:
    raise FileNotFoundError(f"{test_dir} klasöründe hiçbir .jpg veya .png dosyası bulunamadı.")

# Rastgele bir test görseli seç
img_test_path = np.random.choice(val_paths)

# Resmi aç ve RGB'ye çevir
img = Image.open(img_test_path).convert("RGB")

# Gerekirse center crop uygulanabilir ama zaten kullanılan datasetin hepsi 256x256 boyutunda olduğuna göre gerek yok
# img = transforms.CenterCrop(SIZE)(img)

# NumPy array'e çevir
img = np.array(img)

# RGB'den Lab uzayına dönüştür
img_lab = rgb2lab(img).astype("float32")

# Tensöre çevir
img_lab = transforms.ToTensor()(img_lab)

# L kanalını normalize et [-1, 1] aralığına getir
L = img_lab[[0], ...] / 50. - 1.

# Batch dimension ekle
L = L.unsqueeze(0)

# Modeli yükle
generator = UNetGenerator(input_nc=1, output_nc=2,ngf=64)
checkpoint_path = r''#Checpointin kayıtlı olduğu yol
generator.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

# Cihazı ayarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
generator.eval()

# Modelden ab tahmini al
with torch.no_grad():
    ab = generator(L.to(device))

# RGB'ye dönüştür
rgb_out = lab_to_rgb(L.to(device), ab.to(device))

# Görselleştir
plt.figure(figsize=(10, 10))

# Gri görüntü
plt.subplot(1, 2, 1)
plt.imshow(L[0].permute(1, 2, 0), cmap='gray', interpolation='bilinear')
plt.title('Black and White Image')
plt.axis('off')

# Renklendirilmiş görüntü
plt.subplot(1, 2, 2)
plt.imshow(rgb_out[0], interpolation='bilinear')
plt.title('AI Generated Color Image')
plt.axis('off')

plt.tight_layout()
plt.show()
