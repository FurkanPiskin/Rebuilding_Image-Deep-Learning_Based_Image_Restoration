import torch

from utils import normalize_tensor  # Eğer farklı dosyada ise uygun şekilde import et


data = torch.tensor([0.0, 0.5, 1.0])


normalized = normalize_tensor(data, smin=0.0, smax=1.0, tmin=0.0, tmax=255.0)

print("Normalized:", normalized)


expected = torch.tensor([0.0, 127.5, 255.0])


assert torch.allclose(normalized, expected, atol=1e-4), "Normalization yanlış çalışıyor"
print("✅ Normalization doğru çalışıyor!")
