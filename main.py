import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
from model import *
from utils import *

# Constants
dataset_path = 'images'
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE1 = 0.0001
LEARNING_RATE2 = 0.002
HIDDEN_SIZE = 64
LATENT_DIM = 100 # How complex the generated Image is

# Custom Dataset
dataset = CustomDataset(dataset_path)

# Load data during training
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models
discriminator = Discriminator(3, HIDDEN_SIZE)
generator = Generator(LATENT_DIM, 3, HIDDEN_SIZE)

# Load the generator and discriminator models
# generator.load_state_dict(torch.load('generator.pth'))
# discriminator.load_state_dict(torch.load('discriminator.pth'))

# Initialize optimizers
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE1)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE2)

# Binary cross-entropy loss
criterion = nn.BCELoss()


print(f'The Discriminator has {count_parameters(discriminator):,} trainable parameters')
print(f'The Generator has {count_parameters(generator):,} trainable parameters')


for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, leave=True)

    discriminator_loss = 0
    generator_loss = 0

    for idx, x in enumerate(loop):

        batch_size = x.size(0)

        # reset gradients
        optimizer_discriminator.zero_grad()
        optimizer_generator.zero_grad()

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Training discriminator against real images
        real_results = discriminator(x)
        real_loss = criterion(real_results, real_labels)
        real_loss.backward()

        # Training discriminator against fake images
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1)
        fake_images = generator(noise)
        fake_results = discriminator(fake_images)
        fake_loss = criterion(fake_results, fake_labels)
        fake_loss.backward()

        # Optimizer step for the discriminator
        optimizer_discriminator.step()

        optimizer_discriminator.zero_grad()
        optimizer_generator.zero_grad()

        # Training the generator
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1)
        fake_images = generator(noise)
        fake_results = discriminator(fake_images)
        gen_loss = criterion(fake_results, real_labels)
        gen_loss.backward()
        optimizer_generator.step()

        # Save images occasionally
        if epoch % 1 == 0 and idx == 0:
            SavePNG(fake_images, epoch)

        # Add up losses
        discriminator_loss += fake_loss + real_loss
        generator_loss += gen_loss

    average_generator_loss = generator_loss / len(train_loader)
    average_discriminator_loss = discriminator_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Discriminator Loss: {average_discriminator_loss:.4f}"
          f", Generator Loss: {average_generator_loss:.4f}")

    # Occasionally save the model
    if epoch % 5 == 0:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
