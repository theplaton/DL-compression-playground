import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
from torchvision import transforms, datasets
import torch
from torchvision import datasets, transforms
from sklearn.metrics import hamming_loss
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from combined_model import VAE
from data_loader import generate_data_loaders
import itertools
import numpy as np


def read_configs():
    # Specify the path to your JSON file
    json_file_path = './config.json'

    # Read the JSON file into a dictionary
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    return data


def resolve_device():
    # Check for CUDA and MPS availability
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    # Set the device based on availability
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Print the selected device
    print(f"Using device: {device}")
    
    return device


if __name__ == '__main__':

    device = resolve_device()
    
    ################################################################################
    #####################  CREATING DATA LOADERS  ##################################
    ################################################################################

    config = read_configs()
    batch_size = 256
    # Get dataset
    attributes_to_predict = ['Male','Young','Bald','Mustache','Pale_Skin','No_Beard','Receding_Hairline']
    num_attr = len(attributes_to_predict)
    train_loader, test_loader = generate_data_loaders(config['data_path'],config['attr_path'], batch_size, attributes_to_predict, num_workers = 16)
    
    ################################################################################
    ##############################  TRAINING  ######################################
    ################################################################################
    alphas = [0.04]
    latent_dim = 256
    loss_reductions = ['sum']
    for alpha, loss_reduction in itertools.product(alphas,loss_reductions):
        beta = round(1.0 - alpha, 2)
        print(f"Starting training: alpha = {alpha}, beta = {beta}, reduction = {loss_reduction}")
        def loss_function(recon_x, x, mu, log_var, attr, pred_attr):
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
            attribute_loss = F.binary_cross_entropy(pred_attr, attr, reduction = 'sum')
            return 0.66*(alpha*BCE + beta*KLD)+0.34*attribute_loss
            
        if(loss_reduction == 'mean'):
            def loss_function(recon_x, x, mu, log_var, attr, pred_attr):
                BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
                KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
                attribute_loss = F.binary_cross_entropy(pred_attr, attr, reduction = 'mean')
                return 0.66*(alpha*BCE + beta*KLD)+0.34*attribute_loss
        
        
        vae = VAE(image_channels=3, latent_dim=latent_dim, num_attr = num_attr)
        vae.to(device)
        num_epochs = 5
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
       
        log_interval = 1
        print("Starting training....")
        # Training loop
        
        start_time = time.time()
        current_epoch = 0
        total_batches = len(train_loader)
        batches_to_run = total_batches // 2
        batches_to_test = len(test_loader) // 2
        for epoch in range(num_epochs):
            # Training
            vae.train()
            train_loss = 0
            all_attr = []
            all_pred_attr = []
            for batch_idx, data in enumerate(itertools.islice(train_loader,batches_to_run)):
                ################-----load data to gpu----##############
                img, attr = data
                img = img.to(device)
                attr = attr.to(device)
                attr = (attr + 1) // 2
                #print('Labels: ', attr)
                ################-----Get combined model ouput and calculate loss----#############
                recon_batch, mu, log_var, pred_attr = vae(img)
                #print('Predicted_attributes: ', pred_attr)
                loss = loss_function(recon_batch, img, mu, log_var, attr, pred_attr)
                
                ################----Update weights---#############
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if(epoch == 0 and batch_idx ==0):
                    initial_train_loss = train_loss
                    print(f"Initial train loss: ", initial_train_loss)
                ################-----Calculate hamming loss----#############
                predictions = (pred_attr > 0.5).float()
                all_attr.append(attr)
                all_pred_attr.append(predictions)
                print(f'Epoch: {epoch} \tTraining Loss: {train_loss / (batch_idx+1):.6f}',end="\r")
            all_attr = torch.cat(all_attr).cpu().numpy()
            all_pred_attr = torch.cat(all_pred_attr).cpu().numpy()
            train_hamming_loss = hamming_loss(all_attr, all_pred_attr)
            
            # Validation
            vae.eval()
            val_loss = 0
            all_val_attr = []
            all_val_pred_attr = []

            with torch.no_grad():
                for batch_idx, data in enumerate(itertools.islice(test_loader, batches_to_test)):
                    img, attr = data
                    img = img.to(device)
                    attr = attr.to(device)
                    attr = (attr + 1) // 2
                    
                    recon_batch, mu, log_var, pred_attr = vae(img)
                    loss = loss_function(recon_batch, img, mu, log_var, attr, pred_attr)
                    val_loss += loss.item()

                    predictions = (pred_attr > 0.5).float()
                    all_val_attr.append(attr)
                    all_val_pred_attr.append(predictions)
                all_val_attr = torch.cat(all_val_attr).cpu().numpy()
                all_val_pred_attr = torch.cat(all_val_pred_attr).cpu().numpy()
                val_hamming_loss = hamming_loss(all_val_attr, all_val_pred_attr)
                
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss / len(train_loader):.6f} \tValidation Loss: {val_loss / len(test_loader):.6f} \tTrain Hamming Loss: {train_hamming_loss} \tVal Hamming Loss: {val_hamming_loss:.6f}')
       
        torch.save(vae, f'models/sum_male_young_bald/combined_model_{loss_reduction}-loss_{latent_dim}-latent_{alpha}_{beta}.pth')
        end_time = time.time()
        print(f"Training finished. Total training time in minutes: {((end_time-start_time) / 60):1.2f}")
