import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
import model.layers as ml
import model.utils
import matplotlib.pyplot as plt

class SNPatchGANGenerator(nn.Module):
    

    def __init__(self, in_channels: int = 4,
                       leaky_relu_slope : float = 0.2):
        

        super(SNPatchGANGenerator, self).__init__()

        def gated_conv2d(inp: int, out: int, kern: int, strd: int, pad: int, dil: int = 1,
                         act = nn.LeakyReLU(0.2)):
            """
            Verilen parametrelerle GatedConv2d katmanı oluştur
            """
            conv = ml.GatedConv2d(in_channels=inp, out_channels=out, kernel_size=kern,
                        stride=strd, padding=pad, dilation=dil, activation=act)
            return conv

        def gated_upconv2d(inp: int, out: int, kern: int, strd: int, pad: int):
            """
             Verilen parametrelerle GatedUpConv2d katmanı oluştur.
            """
            return ml.GatedUpConv2d(in_channels=inp, out_channels=out, kernel_size=kern,
                                    stride=strd, padding=pad)


        # Coarse network=Kaba aşamaa
        self.coarse = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     
            gated_conv2d(32, 64, 3, 2, 1),              
            gated_conv2d(64, 64, 3, 1, 1),              
            gated_conv2d(64, 128, 3, 2, 1),             
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_conv2d(128, 128, 3, 1, 2, dil=2),     
            gated_conv2d(128, 128, 3, 1, 4, dil=4),     
            gated_conv2d(128, 128, 3, 1, 8, dil=8),     
            gated_conv2d(128, 128, 3, 1, 16, dil=16),   
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_upconv2d(128, 64, 3, 1, 1),           
            gated_conv2d(64, 64, 3, 1, 1),              
            gated_upconv2d(64, 32, 3, 1, 1),            
            gated_conv2d(32, 16, 3, 1, 1),              
            gated_conv2d(16, 3, 3, 1, 1, act=None),    
        )

        # Attention refinement part of the network=İyileştirme ağı
        self.refine_attention = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     
            gated_conv2d(32, 32, 3, 2, 1),              
            gated_conv2d(32, 64, 3, 1, 1),              
            gated_conv2d(64, 128, 3, 2, 1),             
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_conv2d(128, 128, 3, 1, 1),            
            ml.Self_Attn(128),                          
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_conv2d(128, 128, 3, 1, 1),            
        )

        # Tail part for final refinement=Son çıktıyı iyileştirmek için
        self.refine_tail = nn.Sequential(
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_conv2d(128, 128, 3, 1, 1),            
            gated_upconv2d(128, 64, 3, 1, 1),           
            gated_conv2d(64, 64, 3, 1, 1),              
            gated_upconv2d(64, 32, 3, 1, 1),            
            gated_conv2d(32, 16, 3, 1, 1),              
            gated_conv2d(16, 3, 3, 1, 1, act=None),    
        )


    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
       
        # Debug shapes
        #print("Input images shape:", images.shape)
        #print("Input masks shape:", masks.shape)

        # Normalize images
        normalized_images = model.utils.normalize_tensor(images, smin=0, smax=255, tmin=-1, tmax=1)
        #normalized_images=images
        
       
        if masks.dim() == 3:
             masks = masks.unsqueeze(1)  # (B, 1, H, W)
        
        # Create input tensor
        masked_images = normalized_images * (1 - masks)
        input_tensor = torch.cat([masked_images, masks], dim=1)

        # Step 1: Coarse reconstruction
        X_coarse = self.coarse(input_tensor)
        X_coarse = torch.clamp(X_coarse, -1., 1.)
        X_coarse_raw = X_coarse
        X_coarse = masked_images + X_coarse * masks

        # Step 2: Refinement
        X_rec_with_masks = torch.cat([X_coarse, masks], dim=1)
        X_refine_out = self.refine_attention(X_rec_with_masks)
        X_refine_out = self.refine_tail(X_refine_out)
        X_refine_out = torch.clamp(X_refine_out, -1., 1.)

        # Merging refinement with original image
        X_recon_raw = X_refine_out
        X_recon = X_refine_out * masks + normalized_images * (1 - masks)
        
        #print("UnNormalized")
        #print("X_recon range:", X_recon.min().item(), "-", X_recon.max().item())
        #print("X_coarse_raw range:", X_coarse_raw.min().item(), "-", X_coarse_raw.max().item())
        #print("X_recon_raw range:", X_recon_raw.min().item(), "-", X_recon_raw.max().item())

        #print("---------------------------------------------------")

        # Making image out of tensors
        X_recon = model.utils.normalize_tensor(X_recon, smin=-1, smax=1, tmin=0, tmax=255)
        X_coarse_raw = model.utils.normalize_tensor(X_coarse_raw, smin=-1, smax=1, tmin=0, tmax=255)
        X_recon_raw = model.utils.normalize_tensor(X_recon_raw, smin=-1, smax=1, tmin=0, tmax=255)
        #print("Normalized")
        #print("X_recon range:", X_recon.min().item(), "-", X_recon.max().item())
        #print("X_coarse_raw range:", X_coarse_raw.min().item(), "-", X_coarse_raw.max().item())
        #print("X_recon_raw range:", X_recon_raw.min().item(), "-", X_recon_raw.max().item())

        """
        #X_recon = X_recon / 255.0
        #X_coarse_raw = X_coarse_raw / 255.0
        #X_recon_raw = X_recon_raw / 255.0

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(X_recon[0].permute(1, 2, 0).cpu().detach().numpy())
        axs[0].set_title('Final Reconstructed')

        axs[1].imshow(X_coarse_raw[0].permute(1, 2, 0).cpu().detach().numpy())
        axs[1].set_title('Coarse Output')

        axs[2].imshow(X_recon_raw[0].permute(1, 2, 0).cpu().detach().numpy())
        axs[2].set_title('Refined Output')

        for ax in axs:
            ax.axis('off')

        plt.suptitle(f"Step:")
        plt.tight_layout()
        plt.show()
        """

        return X_recon, X_coarse_raw, X_recon_raw