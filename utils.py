import os
from PIL import Image
import matplotlib.pyplot as plt

def SavePNG(generated_images, epoch):
    # Convert pixel values from [-1, 1] to [0, 1] for imshow
    generated_images = (generated_images + 1) / 2

    # Create a grid of images
    fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
    axs = axs.ravel()

    for idx, image in enumerate(generated_images):
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()

        axs[idx].imshow(image_np)
        axs[idx].axis('off')

    # Save the grid as a PNG file
    grid_filename = os.path.join("Created", f"grid_{epoch}.png")
    plt.tight_layout()
    plt.savefig(grid_filename)
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)