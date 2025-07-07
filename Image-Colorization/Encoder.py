import torch.nn as nn

class EncoderBlock(nn.Module):
   
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(EncoderBlock, self).__init__()  # Üst sınıfı (nn.Module) başlat
        
        layers = [
            nn.LeakyReLU(0.2, inplace=True),  # 0.2 negatif eğime sahip Leaky ReLU aktivasyon fonksiyonu
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)  # Aşağı örnekleme (downsampling) için 2D konvolüsyon katmanı
        ]
        
        if use_batchnorm:  # Batch normalization uygulanıp uygulanmayacağını kontrol et
            layers.append(nn.BatchNorm2d(out_channels))  # Batch normalization katmanı ekle
        
        self.block = nn.Sequential(*layers)  # Katmanları sırayla çalışacak şekilde grupla (Sequential blok oluştur)

    def forward(self, x):
        return self.block(x)  # Girdiyi blok içerisinden geçir
