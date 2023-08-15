import torch.nn as nn

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, hidden_dim=64):
        super(Generator, self).__init__()

        """
        Upscale the noise while decreasing the channels with the deconvolution layers. 
        
        Most of the work is done by the convTranspose2d layer
        
        1) It increases spatial dimensions (upsampling) 
        2) Learns features like edges/textures that it applies after being trained properly 
        3) Learns the proper mapping between noise to create a good image
        """
        self.main = nn.Sequential(

            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            # Actually maps out the RGB channels
            nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=4, stride=2, padding=1, bias=False),

            # Converts pixel values to [-1,1]. Normalization similar to transformations in dataset.py
            nn.Tanh()
        )

    def forward(self, x):
        output = self.main(x)
        return output

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator, self).__init__()

        """
        We continue decreasing the 64x64 image until it becomes 1x1 then we flatten
        This is the only way i can think of for this to work since the dimension needs to be [batch_size]
        for the loss function. 
        
        We end with features hidden_dim * 8, but we then decrease to 1 to flatten it completely
        """
        self.main = nn.Sequential(

            nn.Conv2d(image_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)

        return output