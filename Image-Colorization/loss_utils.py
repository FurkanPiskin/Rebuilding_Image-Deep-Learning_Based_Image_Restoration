class AverageMeter:
   
    
    def __init__(self):
        self.reset()   

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
      
        self.count += count  
        self.sum += count * val  
        self.avg = self.sum / self.count  # Updates the average
        
def create_loss_meters():
  
   
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,  
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}     

def update_losses(model, loss_meter_dict, count):
    
  
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)  
        loss_meter.update(loss.item(), count=count)  # Update the AverageMeter with the loss value        
