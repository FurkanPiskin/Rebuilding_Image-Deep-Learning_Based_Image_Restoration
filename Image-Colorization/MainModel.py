import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

from PatchDiscriminator import PatchDiscriminator
from GanLoss import GANLoss
from Unet_Generator import UNetGenerator
from init import init_model


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=1e-4, lr_D=1e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=75.):
        super().__init__()

      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        # Generator ağını başlat(Unet) 
        if net_G is None:
            self.net_G = init_model(UNetGenerator(input_nc=1, output_nc=2, ngf=64),
                                    model_name='Generator', device=self.device)
        else:
            self.net_G = net_G.to(self.device)

        # Discriminator ağını başlat (PatchDiscriminator)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64),
                                model_name='Discriminator', device=self.device)

        # Loss fonksiyonları başlat
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()

        # Optimizers
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

        # Learning rate schedulers
        self.scheduler_G = StepLR(self.opt_G, step_size=30, gamma=0.1)
        self.scheduler_D = StepLR(self.opt_D, step_size=30, gamma=0.1)

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)

        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        # Forward pass
        self.forward()

        # Update discriminator
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # Update generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    def step_scheduler(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
