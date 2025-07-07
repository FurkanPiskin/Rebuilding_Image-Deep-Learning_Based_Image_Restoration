import os
import glob
import re
import numpy as np
from tqdm import tqdm
import torch
import warnings

from loss_utils import create_loss_meters, update_losses
from Visualize_Epoch import visualize
from Log_Results import log_results
from DataLoader import make_dataloaders
from MainModel import MainModel
from Unet_Generator import UNetGenerator
from PatchDiscriminator import PatchDiscriminator

warnings.filterwarnings('ignore')

# Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset yolu(Train)
dataset_path = ""
paths = glob.glob(dataset_path)

if len(paths) == 0:
    raise FileNotFoundError(f"Resim bulunamadı: {dataset_path}")

# Otomatik sayıya göre güncelle
TOTAL_SAMPLES = len(paths)
TRAIN_SAMPLES_PERCENTAGE = 0.9
N_TRAINING_SAMPLES = int(TOTAL_SAMPLES * TRAIN_SAMPLES_PERCENTAGE)

np.random.seed(123)
paths_subset = np.random.choice(paths, TOTAL_SAMPLES, replace=False)

SIZE = 256
np.random.seed(123)
rand_idxs = np.random.permutation(TOTAL_SAMPLES)
train_idxs = rand_idxs[:N_TRAINING_SAMPLES]
val_idxs = rand_idxs[N_TRAINING_SAMPLES:]

train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]

# Model parametreleri
input_nc = 1
output_nc = 2
ngf = 64

# UNet ve Discriminator tanımı
netG = UNetGenerator(input_nc, output_nc, ngf)
discriminator = PatchDiscriminator(input_c=3, n_down=3, num_filters=64)

# Epoch numarasını checkpoint dosya adından al
def get_epoch_from_path(path):
    match = re.search(r'epoch_(\d+)', path)
    if match:
        return int(match.group(1))
    return 0

# Eğitim fonksiyonu
def train_model(model, train_dl, val_dl, epochs, save_every=5, resume_path=None):
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        model.net_G.load_state_dict(torch.load(resume_path, map_location=device))
        start_epoch = get_epoch_from_path(resume_path)
        print(f"Starting from epoch {start_epoch + 1}")
    else:
        print("No valid checkpoint provided. Training from scratch.")
        start_epoch = 0

    for e in range(start_epoch, epochs):

        # 🔸 Learning rate'leri yazdır
        print(f"\nEpoch {e + 1}/{epochs} Learning Rates:")
        for name, optimizer in [('G', model.opt_G), ('D', model.opt_D)]:
            for i, group in enumerate(optimizer.param_groups):
                print(f"{name} optimizer group {i} LR: {group['lr']}")

        loss_meter_dict = create_loss_meters()
        model.train()
        last_data = None

        for data in tqdm(train_dl, desc=f"Epoch {e + 1}/{epochs}"):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            last_data = data

        print(f"\nEpoch {e + 1}/{epochs} Losses:")
        log_results(loss_meter_dict)

        if last_data is not None:
            visualize(model, last_data, save=True)
        else:
            print("Warning: No data to visualize for this epoch!")

        # Validation görselleştirme
        model.eval()
        with torch.no_grad():
            for val_data in val_dl:
                model.setup_input(val_data)
                model.forward()
                visualize(model, val_data, save=True)
                break

        # Checkpoint kaydetme
        if (e + 1) % save_every == 0:
            os.makedirs("./Generator_Checkpoints", exist_ok=True)
            checkpoint_path = f"./Generator_Checkpoints/generator_epoch_{e + 1}.pth"
            torch.save(model.net_G.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # 🔸 Belirli epochlarda scheduler çalıştır
        if (e + 1) in [60, 75]:
            print(f"-> Learning rate scheduler triggered at epoch {e + 1}")
            model.step_scheduler()

    return loss_meter_dict


# Ana çalışma bloğu
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    train_dl = make_dataloaders(paths=train_paths, crop_size=SIZE)
    val_dl = make_dataloaders(paths=val_paths, crop_size=SIZE)

    model = MainModel()

    # Buraya kendi checkpoint yolunu yaz, örnek:
    resume_checkpoint_path = "C:\\Users\\bozku\\Desktop\\BİTİRME\\Generator_Checkpoints\\generator_epoch_64.pth"
    # Eğer sıfırdan başlamak istersen resume_path=None bırakabilirsin

    loss_meter_dict_output = train_model(
        model,
        train_dl,
        val_dl,
        epochs=90,
        save_every=1,
        resume_path=resume_checkpoint_path
    )

    """
    Loss_D_real: D'nin gerçek veriyi tanıma başarısı
    loss_D_fake: D'nin sahte veriyi tanıma başarısı
    loss_D: D'nin genel başarısı
    loss_G_GAN: G'nin sahteyi gerçek gibi yapma başarısı
    loss_G_L1: G'nin gerçek renklere yakınlığı
    loss_G: G'nin toplam hatası
    """
