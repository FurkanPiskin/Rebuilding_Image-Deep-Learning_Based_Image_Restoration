


import os
import torch
import pytorch_lightning as pl
import sys
import os
from pytorch_lightning.callbacks import ModelCheckpoint


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from model.gan import SNPatchGAN
from model.data import PlacesDataModule
import model.gan as gan
import model.data as loader

import torch



def train():
    pl.seed_everything(42)

    # Model ve veri
    model = SNPatchGAN()
    data = PlacesDataModule()

    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",                 # Kaydedilecek klasör
    filename="epoch-{epoch:03d}",         # Dosya adı formatı
    save_top_k=-1,                        # Tüm epoch'ları kaydet
    every_n_epochs=1                      # Her epoch'ta kaydet
)

    # Trainer 
    trainer = pl.Trainer(
    accelerator="gpu",   # GPU kullanımı
    callbacks=[checkpoint_callback],
    devices=1,           # kaç GPU kullanılacak
    max_epochs=40,
    val_check_interval=5475,
    logger=True,
    enable_checkpointing=True
    )
    resume_path = ""

    # Eğitimi başlat
    #trainer.fit(model, data)

     # Eğitimi başlat (checkpoint varsa oradan başla)
    trainer.fit(model, datamodule=data,ckpt_path=resume_path)

    #  Modeli manuel olarak kaydet
    # .pth uzantısıyla sadece ağırlıkları (state_dict) kaydederiz
    save_path = "trained_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model başarıyla kaydedildi: {save_path}")


if __name__ == '__main__':
    train()
