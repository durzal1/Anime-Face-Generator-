import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchtext.data.utils import get_tokenizer


class CustomDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a consistent size
        transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

        self.image_names = [name for name in os.listdir(image_dir)]


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        # Acquire information
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)

        # Create tensor of (3,64,64) matrix of image
        image = Image.open(image_path).convert("RGB")

        # Apply transformation
        image = self.transform(image)

        return image