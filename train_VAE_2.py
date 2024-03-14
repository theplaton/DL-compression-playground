#!/usr/bin/env python
# coding: utf-8

# In[1]:
def divide_by_256(x):
        return x / 256
if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
    from VAE import VAE
    
    
    # In[2]:
    batch_size = 32
    
    torch.cuda.is_available()
    device = torch.device("cuda")
    print(device)
    
    
    # In[9]:
    
    ################################################################################
    #####################  CREATING DATA LOADERS  ##################################
    ################################################################################
    from ImageLoader import CustomImageDataset
    
    # Define a custom transformation that divides each pixel by 256

    
    #Define a transform to resize the images, convert them to tensors, and scale to [0, 1]
    transform = transforms.Compose([
        # transforms.Resize((144, 158)),
        transforms.ToTensor(),
        transforms.Lambda(divide_by_256)           # Divide each pixel by 256
    ])
    
    # Create the dataset
    #train_loader, test_loader = generate_data_loaders('/Users/Nate/Documents/cs583/midterm/data/img_align_celeba/img_align_celeba')
    dataset = CustomImageDataset(root_dir='/Users/Nate/Documents/cs583/midterm/data/img_align_celeba/img_align_celeba', transform = transform)
    print("Dataset loaded")
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    #train_loader, test_loader = generate_data_loaders('/Users/Nate/Documents/cs583/midterm/data/img_align_celeba/img_align_celeba')
    
    
    ################################################################################
    ##############################  TRAINING  ######################################
    ################################################################################
    
    vae = VAE(image_channels=3, latent_dim=128)
    vae.to(device)
    
    num_epochs = 10
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    def loss_function(recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    log_interval = 1
    print("Starting training....")
    # Training loop
    
    
    # In[ ]:
    
    
    start_time = time.time()
    current_epoch = 0
    total_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        # Training
        vae.train()
        batches_processed = 0
        train_loss = 0
        current_epoch += 1
        remaining_epochs = num_epochs - current_epoch
        #train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Train Loss: {0:.6f}", leave = False)
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
        ################################################################################
        ##################### CALCULATING PROCESSING TIME  #############################
        ################################################################################
            batches_processed += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            processing_rate = batches_processed / elapsed_time
            remaining_batches = total_batches - batches_processed
            total_remaining_batches = remaining_epochs * total_batches + remaining_batches
            total_remaining_time = total_remaining_batches / processing_rate
            print(f"Current loss: {train_loss / (batch_idx+1):8.4f}, Remaining time (s): {total_remaining_time:8.2f}, Remaining batches: {total_remaining_batches:5d}, Remaining_images: {total_remaining_batches*32:5d}, Processing_rate: {processing_rate:6.2f}, BatchID: {batch_idx}", end="\r")
        ################################################################################
            
            # data = data * 2 - 1  # Rescale images from [0, 1] to [-1, 1]
            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(f'batch: {batch_idx} \t{len(train_loader)}')
            #train_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / (batch_idx+1):.6f}")
            #print(f"Current loss: {train_loss / (batch_idx+1)}, Remaining time (s): {remaining_time}", end="\r")
        
        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device)
                #data = data * 2 - 1  #Rescale images from [0, 1] to [-1, 1]
                recon_batch, mu, log_var = vae(data)
                loss = loss_function(recon_batch, data, mu, log_var)
                val_loss += loss.item()
    
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss / len(train_loader):.6f} \tValidation Loss: {val_loss / len(test_loader):.6f}')


# In[ ]:





# In[ ]:




