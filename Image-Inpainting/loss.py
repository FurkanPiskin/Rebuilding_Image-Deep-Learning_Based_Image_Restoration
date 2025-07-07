"""
Various losses for GAN training process.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import vgg16
from torchvision.transforms import Normalize
import model.utils

# ===================== [CODE] =====================


class GeneratorLoss(nn.Module):
    

    def forward(self, X_fake: torch.Tensor) -> torch.Tensor:
        return (-1) * torch.mean(X_fake)




class DiscriminatorLoss(nn.Module):
    

    def __init__(self, real_smooth=1.0, fake_smooth=-1.0):
        super().__init__()
        self.real_smooth = real_smooth
        self.fake_smooth = fake_smooth

    def forward(self, X_real: torch.Tensor, X_fake: torch.Tensor) -> torch.Tensor:
        
        real_loss = torch.mean(F.relu(self.real_smooth - X_real))
        fake_loss = torch.mean(F.relu(self.fake_smooth + X_fake))
        return real_loss + fake_loss

        



class ReconLoss(torch.nn.Module):

    

    def __init__(self):
        super(ReconLoss, self).__init__()
        self.recon_inmask_alpha  = 1 / 40
        self.recon_outmask_alpha = 1 / 40
        self.coarse_inmask_alpha  = 1 / 40
        self.coarse_outmask_alpha = 1 / 40


    def loss(self, images_1: torch.Tensor,
                   images_2: torch.Tensor,
                   masks: torch.Tensor,
                   coef: float,
                   mask_weight: float = 5.0) -> torch.Tensor:

        masks_bit_ratio = torch.mean(masks.view(masks.size(0), -1), dim=1)
        masks_bit_ratio = masks_bit_ratio.view(-1, 1, 1, 1)
        masks = torch.unsqueeze(masks, dim=3)
        #print(f'images1_Shape:{images_1.shape}')
        #print(f'images2_Shape:{images_2.shape}')
        masks=masks.permute(0,3,1,2)
        #print(f'masks_Shape_loss:{masks.shape}')

        #masked_diff = torch.abs(images_1 - images_2) * masks / masks_bit_ratio
        #loss = coef * torch.mean(masked_diff)

        masked_diff = torch.abs(images_1 - images_2) * masks * mask_weight / masks_bit_ratio

    # Maskelenmemiş bölge için normal fark
        unmasked_diff = torch.abs(images_1 - images_2) * (1 - masks) / (1 - masks_bit_ratio)

        loss = coef * (torch.mean(masked_diff) + torch.mean(unmasked_diff))
        return loss
        


    def forward(self, images: torch.Tensor,
                      coarse_images: torch.Tensor,
                      recon_images: torch.Tensor,
                      masks: torch.Tensor) -> torch.Tensor:
        
        recon_inmsak_loss  = self.loss(images, recon_images, masks, self.recon_inmask_alpha, mask_weight=5.0)
        recon_outmask_loss = self.loss(images, recon_images, 1 - masks, self.recon_outmask_alpha, mask_weight=1.0)
        coarse_inmask_loss = self.loss(images, coarse_images, masks, self.coarse_inmask_alpha, mask_weight=5.0)
        coarse_outmask_loss = self.loss(images, coarse_images, 1 - masks, self.coarse_outmask_alpha, mask_weight=1.0)
        return recon_inmsak_loss + recon_outmask_loss + coarse_inmask_loss + coarse_outmask_loss

class VGGPerceptualLoss(nn.Module):  

    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    def forward(self, input, target):
        input = self.normalize(input)
        target = self.normalize(target)
        input_feats = self.vgg(input)
        target_feats = self.vgg(target)
        return F.l1_loss(input_feats, target_feats)
    

class TVLoss(nn.Module):
    def forward(self, x):
        return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))    

class VGGStyleLoss(nn.Module):
    def __init__(self, device):
        super(VGGStyleLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    def gram_matrix(self, feature):
        (b, c, h, w) = feature.size()
        features = feature.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, input, target):
        input = self.normalize(input)
        target = self.normalize(target)
        input_feats = self.vgg(input)
        target_feats = self.vgg(target)

        input_gram = self.gram_matrix(input_feats)
        target_gram = self.gram_matrix(target_feats)

        return F.l1_loss(input_gram, target_gram)    