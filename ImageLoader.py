import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from VAE import VAE


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Get the dimensions of the image
        width, height = image.size

        # Define the coordinates of the box to crop
        # (left, upper, right, lower)
        crop_box = (9, 13, width - 9, height - 13)

        # 178 -> 160
        # 218 -> 192

        # Crop the image
        cropped_image = image.crop(crop_box)

        if self.transform:
            final_image = self.transform(cropped_image)
        return final_image
