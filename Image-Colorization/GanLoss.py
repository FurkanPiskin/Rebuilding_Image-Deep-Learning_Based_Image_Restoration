import torch
import torch.nn as nn
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        
        super().__init__()
        
        # Register buffers for real and fake labels. Buffers are tensors that are not updated by optimization but are saved and loaded.
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        
        # Choose the loss function based on the specified GAN mode
        if gan_mode == 'vanilla':
            # Vanilla GAN uses binary cross-entropy loss
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            # Least-Squares GAN uses mean squared error loss
            self.loss = nn.MSELoss()
        
        # You can add more modes like 'wgan-gp' here if needed

    def get_labels(self, preds, target_is_real):
        
        # Determine the appropriate label based on whether the target is real or fake
        if target_is_real:
            labels = self.real_label  # Assign real label if the target is real
        else:
            labels = self.fake_label  # Assign fake label if the target is fake

        # Expand the labels to match the shape of the predictions tensor
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        
        labels = self.get_labels(preds, target_is_real)  # Get the appropriate labels
        loss = self.loss(preds, labels)  # Compute the loss
        return loss