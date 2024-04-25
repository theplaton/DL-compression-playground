import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
import time
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from vae_2 import VAE
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


to_image = transforms.ToPILImage()
transform = transforms.Compose([
        transforms.ToTensor()
    ])

def plot_model_reconstructions(model,image1, image2):
    width, height = image1.size
    crop_box = (9, 13, width - 9, height - 13)
    cropped_image1 = image1.crop(crop_box)
    cropped_image2 = image2.crop(crop_box)
    device = torch.device("cuda")
    transformed_image1 = transform(cropped_image1).to(device)
    transformed_image2 = transform(cropped_image2).to(device)
    
    
    batch_tensor = torch.stack([transformed_image1, transformed_image2], dim=0)
    model.eval()
    reconstructed_img, mu, logvar, attr= model(batch_tensor)
    rec_image1 = to_image(reconstructed_img[0])
    rec_image2 = to_image(reconstructed_img[1])
    attrs = attr.detach().cpu().numpy()
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
    axs[0, 0].imshow(cropped_image1)
    axs[0, 0].axis('off')
    axs[1, 0].imshow(cropped_image2)
    axs[1, 0].axis('off')
    axs[0, 1].imshow(rec_image1)
    axs[0, 1].axis('off')
    axs[1, 1].imshow(rec_image2)
    axs[1, 1].axis('off')
    plt.tight_layout(h_pad=0.1,w_pad = 0.4)
    plt.show()
    # Show the image
    # display(cropped_image1)
    # display(cropped_image2)
    # display(rec_image1)
    # display(rec_image2)
    return attrs

def plot_images_from_models(models_dir, alphas, latent_dim, loss_reduction, cropped_image1, cropped_image2, device, round_to):
    # Setup plot
    fig, axs = plt.subplots(nrows=2, ncols=len(alphas)+1, figsize=(10, 4))  # nrows is number of images, ncols is number of models
    # display(cropped_image1)
    # display(cropped_image2)

    transformed_image1 = transform(cropped_image1).to(device)
    transformed_image2 = transform(cropped_image2).to(device)
    batch_tensor = torch.stack([transformed_image1, transformed_image2], dim=0)
    axs[0, 0].imshow(cropped_image1)
    axs[1, 0].imshow(cropped_image2)
    axs[0, 0].axis('off')
    axs[1, 0].axis('off')
    for i, alpha in enumerate(alphas):
        beta = round(1 - alpha,round_to)
        model_path = f'{models_dir}/combined_model_{loss_reduction}-loss_{latent_dim}-latent_{alpha}_{beta}.pth'
        model = torch.load(model_path)
        model.to(device)
        model.eval()
    
        # Generate reconstructed images
        reconstructed_imgs, mu, logvar, attr = model(batch_tensor)
        
        rec_image1 = to_image(reconstructed_imgs[0].cpu())
        rec_image2 = to_image(reconstructed_imgs[1].cpu())
    
        # Display images
        axs[0, i+1].imshow(rec_image1)
        axs[1, i+1].imshow(rec_image2)
        axs[0, i+1].set_title(f"latent: {latent_dim}, alpha: {alpha}", fontsize = 4)
        axs[0, i+1].axis('off')
        axs[1, i+1].axis('off')
    # Show the plot
    plt.tight_layout(h_pad=0.1,w_pad = 0.4)
    plt.show()


def plot_generated_images(models_dir, alphas, latent_dim, loss_reduction, num_samples, device, round_to):
    fig, axs = plt.subplots(nrows=num_samples, ncols=len(alphas), figsize=(10, 9))  # Adjust figsize as needed
    for idx, alpha in enumerate(alphas):
        beta = round(1 - alpha, round_to)
        model_path = f'{models_dir}/combined_model_{loss_reduction}-loss_{latent_dim}-latent_{alpha}_{beta}.pth'
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        model_name = f'{loss_reduction}, a={alpha}, l={latent_dim}'
        
        z = torch.randn(num_samples, latent_dim)
        z = z.to(next(model.parameters()).device)
        with torch.no_grad():
            generated_data = model.decoder(z)
        axs[0,idx].set_title(f"{model_name}", fontsize = 6)    
        for i, img in enumerate(generated_data):
            img = to_image(img.cpu())  # Convert tensor to PIL Image and move to CPU
            
            axs[i,idx].imshow(img)
            axs[i,idx].axis('off')
    
    # Show the plot
    plt.tight_layout(h_pad=0.1,w_pad = 0.1)
    plt.show()