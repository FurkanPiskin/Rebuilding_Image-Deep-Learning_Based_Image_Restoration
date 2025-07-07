import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import model.data as loader
import model.utils as u
import matplotlib.pyplot as plt
import model.loss
import model.generator
import model.discriminator
import random

def safe_psnr(img1, img2, data_range=255):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((data_range ** 2) / mse)

class SNPatchGAN(pl.LightningModule):
    def __init__(self):
        super(SNPatchGAN, self).__init__()
       
        
        self.generator = model.generator.SNPatchGANGenerator()
        self.discriminator = model.discriminator.SNPatchGANDiscriminator()

        self.g_loss_fn = model.loss.GeneratorLoss()
        self.d_loss_fn = model.loss.DiscriminatorLoss(real_smooth=0.9, fake_smooth=-0.9)
        self.r_loss_fn = model.loss.ReconLoss()
        #self.style_loss_fn=model.loss.VGGStyleLoss(self.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg_perceptual_loss = model.loss.VGGPerceptualLoss(device=device)
        
        self.tv_loss=model.loss.TVLoss()

        # Important: manual optimization
        self.automatic_optimization = False

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        return self.generator(images, masks)
   
    def on_train_epoch_start(self):
        opt_g, opt_d = self.optimizers()
        lr_g = opt_g.param_groups[0]['lr']
        lr_d = opt_d.param_groups[0]['lr']
        print(f"[Epoch {self.current_epoch}] Generator LR: {lr_g:.6f}, Discriminator LR: {lr_d:.6f}")

        # Eğer belirli epoch'ta LR düşürmek istiyorsan:
        if self.current_epoch in [20,25,30]:
            #new_lr_g = lr_g * 3e-5
            #new_lr_d = lr_d * 1e-5

            new_lr_g =3e-5
            new_lr_d =1e-5

            for param_group in opt_g.param_groups:
                param_group['lr'] = new_lr_g
            for param_group in opt_d.param_groups:
                param_group['lr'] = new_lr_d

            print(f"Learning rate updated at epoch {self.current_epoch} → Generator: {new_lr_g}, Discriminator: {new_lr_d}")

      

    
    def on_train_epoch_end(self):
        # Görselleştirme için bir örnek görsel belirle
        image_path = '' #Örnek görsel yolu
        save_dir = "epoch_outputs"
        os.makedirs(save_dir, exist_ok=True)
        self.visualize_example(image_path, self.current_epoch, save_dir)

    def load_fixed_random_mask(image_shape, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        height, width = int(image_shape[0]), int(image_shape[1])
        mask = loader.generate_random_mask(width=width, height=height)  # PIL mask
        mask = np.array(mask).astype(np.float32)
        return torch.from_numpy(mask) 
        
    def visualize_example(self, image_path, epoch, save_dir):
        device = self.device
        origin_image = u.load_image(image_path).to(device)
        origin_image = origin_image.permute(2, 0, 1).unsqueeze(0)

        image_shape = origin_image.shape[2:]
        origin_mask = u.load_random_mask(image_shape).to(device).unsqueeze(0).unsqueeze(0)
        #origin_mask = loader.generate_random_mask(image_shape).to(device).unsqueeze(0).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            gan_output, _, _ = self(origin_image, origin_mask)

        gan_output = gan_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gan_output = (gan_output / 255.0).clip(0, 1)

        plt.imshow(gan_output)
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"{save_dir}/epoch_{epoch:03d}.png")
        plt.close()

    def configure_optimizers(self):
    # Generator optim
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        # Discriminator optim
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999))

        # StepLR scheduler (10 epoch'ta bir LR yarıya iniyor)
        #g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.5)
        #d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=10, gamma=0.5)

        return [g_optimizer, d_optimizer]


    """
    def on_train_start(self):
        # İstediğin learning rate değerleri
        new_lr_g = 3e-4
        new_lr_d = 5e-6

        opt_g, opt_d = self.optimizers()

        for param_group in opt_g.param_groups:
            param_group['lr'] = new_lr_g

        for param_group in opt_d.param_groups:
            param_group['lr'] = new_lr_d

        print(f"Learning rates set to Generator: {new_lr_g}, Discriminator: {new_lr_d}")
       
    """
    def training_step(self, batch, batch_idx):
        #current_lr = self.optimizers()[0].param_groups[0]['lr']
        #print(f"[Epoch {self.current_epoch}] Generator LR: {current_lr:.6f}")
        images, masks = batch
        batch_size = images.size(0)

        fake_images, coarse_raw, recon_raw = self(images, masks)

        # Get optimizers
        opt_g, opt_d = self.optimizers()

        # ---------------------
        # 1. Discriminator step
        # ---------------------
        self.toggle_optimizer(opt_d)

        all_images = torch.cat([fake_images.detach(), images], dim=0)
        all_masks = torch.cat([masks, masks], dim=0)
        all_outputs = self.discriminator(all_images, all_masks)

        fake_output = all_outputs[:batch_size]
        real_output = all_outputs[batch_size:]

        d_loss = self.d_loss_fn(real_output, fake_output)
        self.log('d_loss', d_loss, prog_bar=True)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.untoggle_optimizer(opt_d)

        # ---------------------
        # 2. Generator step
        # ---------------------
        self.toggle_optimizer(opt_g)

        d_output = self.discriminator(fake_images, masks)
        g_loss = self.g_loss_fn(d_output)
        r_loss = self.r_loss_fn(images, coarse_raw, recon_raw, masks)
        perceptual_loss = self.vgg_perceptual_loss(fake_images, images)
        # 4. Total Variation Loss
        tv_loss = self.tv_loss(fake_images)

        #style_loss=self.style_loss_fn(fake_images,images)
        #total_g_loss = g_loss + r_loss + 0.1 * perceptual_loss + 0.1 * tv_loss
        #total_g_loss = 1.0 * g_loss + 6.0 * r_loss + 0.5 * perceptual_loss + 0.05 * tv_loss+self.style_w*style_loss

        #Aşağıdaki loss ağırlıkları eğitimin çıktı kalitesine göre belirlenip kullanılır.
        """
        total_g_loss = (
        1.0 * g_loss + 
        10.0 * r_loss + 
        0.5 * perceptual_loss + 
        0.01 * tv_loss  
        )
        """ 

        """
        total_g_loss = (
        1.0 * g_loss + 
        7.0 * r_loss + 
        1.0 * perceptual_loss + 
        0.05 * tv_loss
)
        """
        
        total_g_loss = (
        1.0 * g_loss + 
        8.0 * r_loss + 
        0.75 * perceptual_loss + 
        0.03 * tv_loss
        )
        
        
        """
        total_g_loss = (
        1.0 * g_loss + 
        10.0 * r_loss + 
        0.5 * perceptual_loss + 
        0.01 * tv_loss + 
        0.5 * style_loss
        )
        """
        """
        total_g_loss = (
        1.0 * g_loss + 
        6.0 * r_loss + 
        0.5 * perceptual_loss + 
        0.05 * tv_loss + 
        5.0 * style_loss   
        )
        """

        """
        total_g_loss = (
        1.0 * g_loss + 
        3.0 * r_loss + 
        0.7 * perceptual_loss + 
        0.1 * tv_loss + 
        10.0 * style_loss
        )
        """ 
        

        

        
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('r_loss', r_loss, prog_bar=True)
        self.log('vgg_loss', perceptual_loss, prog_bar=True)#new
        self.log('tv_loss', tv_loss, prog_bar=True)#new
        #self.log('style_loss',  style_loss,  prog_bar=True)

        opt_g.zero_grad()
        self.manual_backward(total_g_loss)
        opt_g.step()

        self.untoggle_optimizer(opt_g)
        #print(f"Model device: {next(self.parameters()).device}")
        #print(f"Image device: {images.device}, Mask device: {masks.device}")

    


    def validation_step(self, batch, batch_idx):
        images, masks = batch

        with torch.no_grad():
            fake_images, _, _ = self(images, masks)

        fake_np = fake_images.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
        real_np = images.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

        fake_np = (fake_np * 255).astype(np.uint8)
        real_np = (real_np * 255).astype(np.uint8)

        batch_psnr, batch_ssim = [], []

        for i in range(real_np.shape[0]):
            r = real_np[i]
            f = fake_np[i]

            min_side = min(r.shape[0], r.shape[1])
            win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)

            batch_psnr.append(safe_psnr(r, f, data_range=255))
            batch_ssim.append(ssim(r, f, data_range=255, channel_axis=2, win_size=win_size))

        avg_psnr = np.mean(batch_psnr)
        avg_ssim = np.mean(batch_ssim)

        self.log('val_psnr', avg_psnr, prog_bar=True)
        self.log('val_ssim', avg_ssim, prog_bar=True)

        # Görsel kaydetmek için sadece ilkini al
        self.val_outputs.append({
            'fake_image': fake_images[0].detach().cpu(),
            'real_image': images[0].detach().cpu(),
            'mask': masks[0].detach().cpu()
        })


    def on_validation_epoch_start(self):
        self.val_outputs = []


    def on_validation_epoch_end(self):
        if len(self.val_outputs) == 0:
            return

        batch = random.choice(self.val_outputs)
        fake_img = batch['fake_image']
        real_img = batch['real_image']
        mask = batch['mask']

        save_dir = "validation_outputs"
        os.makedirs(save_dir, exist_ok=True)

        vutils.save_image(fake_img, os.path.join(save_dir, f"fake_epoch{self.current_epoch}.png"), normalize=True)
        vutils.save_image(real_img, os.path.join(save_dir, f"real_epoch{self.current_epoch}.png"), normalize=True)
        vutils.save_image(mask, os.path.join(save_dir, f"mask_epoch{self.current_epoch}.png"), normalize=True)

        print(f"[VAL] Görseller kaydedildi: epoch {self.current_epoch}")

        self.val_outputs = []

