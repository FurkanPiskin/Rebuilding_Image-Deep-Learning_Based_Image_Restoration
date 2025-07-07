import torch.nn as nn

class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(DecoderBlock, self).__init__()  # Üst sınıfı (nn.Module) başlat
        
        layers = [
            nn.ReLU(inplace=True),  # Doğrusal olmayanlık için ReLU aktivasyon fonksiyonu
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Yukarı örnekleme (upsampling) için transpoze konvolüsyon
            nn.BatchNorm2d(out_channels)  # Batch normalization katmanı
        ]
        
        if use_dropout:  # Dropout uygulanıp uygulanmayacağını kontrol et
            layers.append(nn.Dropout(0.5))  # %50 oranında dropout katmanı ekle
        
        self.block = nn.Sequential(*layers)  # Katmanları sırayla çalışacak şekilde grupla (Sequential blok oluştur)

    def forward(self, x):
        """Decoder bloğu üzerinden ileri besleme işlemi."""
        return self.block(x)  # Girdiyi blok içerisinden geçir
