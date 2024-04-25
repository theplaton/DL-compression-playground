from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import pandas as pd
import sys


# def divide_by_255(x): -- unneeded
#     return x / 255

class CustomImageDataset(Dataset):
    def __init__(self, root_dir,attr_dir, transform, attr_to_pred):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.attributes = pd.read_csv(attr_dir, delim_whitespace = True, index_col = 0)[attr_to_pred]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        attrs = self.attributes.loc[self.image_files[idx]]
        attrs = torch.tensor(attrs.values, dtype=torch.float32)
        # Get the dimensions of the image
        width, height = image.size

        # Define the coordinates of the box to crop
        # (left, upper, right, lower)
        # 178 -> 160
        # 218 -> 192
        crop_box = (9, 13, width - 9, height - 13)

        # Crop the image
        cropped_image = image.crop(crop_box)

        if self.transform:
            final_image = self.transform(cropped_image)

        return final_image, attrs



def generate_data_loaders(image_dir,attr_dir, batch_size, attr_to_pred, split_ratio=0.8, num_workers=8, shuffle=True, ):
    # Define your transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(divide_by_255)  # scale to 0-1 range ---- ToTensor already does this 
    ])

    # Create the dataset
    dataset = CustomImageDataset(image_dir,attr_dir, transform, attr_to_pred)
    # Split the dataset into train and test sets
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing with multiple workers
    print(f"Training set size: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,  pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    
    return train_loader, test_loader

