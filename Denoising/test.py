import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image, amount=0.05, s_vs_p=0.5):
    noisy = image.clone()
    c, h, w = image.shape
    num_salt = int(amount * h * w * s_vs_p)
    num_pepper = int(amount * h * w * (1.0 - s_vs_p))

    # Salt (white) noise  
    coords = [torch.randint(0, h, (num_salt,)), torch.randint(0, w, (num_salt,))]
    noisy[:, coords[0], coords[1]] = 1.0

    # Pepper (black) noise 
    coords = [torch.randint(0, h, (num_pepper,)), torch.randint(0, w, (num_pepper,))]
    noisy[:, coords[0], coords[1]] = 0.0

    return noisy
def show_rgb_with_bw_noise(image_path, amount=0.05):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    noisy_tensor = add_salt_pepper_noise(image_tensor, amount)

    noisy_np = noisy_tensor.permute(1, 2, 0).numpy()

    plt.imshow(noisy_np)
    plt.title("Salt & Pepper Noise (Black & White on RGB)")
    plt.axis("off")
    plt.show()


show_rgb_with_bw_noise("./img/9.png", amount=0.05)