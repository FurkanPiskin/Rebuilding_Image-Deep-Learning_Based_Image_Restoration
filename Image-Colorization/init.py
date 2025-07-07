from torch import nn
def init_weights(net, init='norm', gain=0.02, name='Generator'):
    
    
    def init_func(m):
        
        classname = m.__class__.__name__ 
        if hasattr(m, 'weight') and 'Conv' in classname:
            # Initialize convolutional layers
            if init == 'norm':
                # Normal distribution initialization
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                # Xavier initialization
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                # Kaiming initialization
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                # Initialize biases to zero
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            # Initialize BatchNorm2d layers
            nn.init.normal_(m.weight.data, 1., gain)  
            nn.init.constant_(m.bias.data, 0.)  
            
    net.apply(init_func)  # Apply the initialization function to all layers in the network
    print(f"{name.capitalize()} model initialized with {init} initialization")

    return net

def init_model(model, model_name, device):
    model = model.to(device)
    model = init_weights(model, name=model_name)
    return model    
"""
Sinir ağlarının eğitim sürecinde ağırlıkların (weights) uygun şekilde başlatılması, modelin doğru ve verimli bir şekilde öğrenmesi için kritik bir adımdır. Eğer ağırlıklar rastgele ve kötü başlatılırsa:

Modelin öğrenmesi yavaşlar veya hiç öğrenemez.

Vanishing (yok olan) ya da exploding (patlayan) gradyan problemleri oluşabilir.

Özellikle derin yapılar veya GAN (Generative Adversarial Network) gibi hassas modeller, uygun başlatma olmadan dengesiz çalışır.

Bu fonksiyon, modeldeki her bir katmanı gezerek uygun ağırlık başlatma yöntemini uygular. Desteklenen başlatma türleri:

'norm': Sıfır ortalama ve belirli standart sapma ile normal dağılım kullanır.

'xavier': Aktivasyonların katmanlar arasında dengeli kalmasını sağlar. Derin ağlarda sık kullanılır.

'kaiming': ReLU veya LeakyReLU gibi aktivasyonlar ile uyumlu şekilde daha derin ağlarda kullanılır.

Ek olarak, bu fonksiyon:

BatchNorm katmanlarının ağırlık ve bias değerlerini de başlatır.

Modelin hangi yöntemle başlatıldığını terminalde kullanıcıya bildirir.

Sonuç olarak init_weights, modelin stabil, hızlı ve etkili şekilde eğitilebilmesi için temel bir hazırlık adımıdır ve her eğitim sürecinin başlangıcında mutlaka uygulanmalıdır.


"""