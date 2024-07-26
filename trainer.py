import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import glob
from torch.utils.data import DataLoader, random_split, ConcatDataset
from crepe import CREPE
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from dataset import MIR1KDataset, Back10Dataset, NSynthDataset

import wandb

def train_epoch(model, dataloader, criterion, optimizer, sr, device):
    model.train()
    running_loss = 0.0
    for audio, labels in tqdm(dataloader):
        audio = audio.to(device)
        labels = labels.to(device)
        labels = F.interpolate(labels.unsqueeze(0), size = audio.shape[-1], mode='linear', align_corners = False).squeeze(0)
        labels = frequency_to_cents(labels)
        label_activations = get_activation_from_label(labels)
        
        optimizer.zero_grad()
        model_activations = model.get_activation(audio, sr)
        
        loss = criterion(model_activations, label_activations)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * audio.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, sr, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(device)
            labels = labels.to(device)
            labels = F.interpolate(labels.unsqueeze(0), size = audio.shape[-1], mode='linear', align_corners = False).squeeze(0)
            labels = frequency_to_cents(labels)
            label_activations = get_activation_from_label(labels)
        
            model_activations = model.get_activation(audio, sr)
            
            loss = criterion(model_activations, label_activations)
            
            running_loss += loss.item() * audio.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    batch_size = 1
    num_epoch = 5000
    num_batches_per_epoch = 8
    max_epochs_without_improvement = 32
    learning_rate = 0.0002
    model_capacity = 'tiny'
    sr = 16000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = CREPE(model_capacity=model_capacity).to(device)
    
    dataset = MIR1KDataset(root_dir="./dataset/MIR-1K")
#    back10 = Back10Dataset(root_dir="./dataset/bach10")
#    nsynth = NSynthDataset(root_dir="./dataset/Nsynth-mixed")
    
#    dataset = ConcatDataset([mir_1k, back10, nsynth])
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Crepe tiny",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "MIR-1K",
        "epochs": num_epoch,
        }
    )

    for epoch in range(1, num_epoch):
        train_loss = 0.0
        for _ in range(num_batches_per_epoch):
            train_loss += train_epoch(model, train_loader, criterion, optimizer, sr, device)
        train_loss /= num_batches_per_epoch
        
        val_loss = validate_epoch(model, val_loader, criterion, sr, device)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'crepe_{model_capacity}_best.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= max_epochs_without_improvement:
            print("Stopping early due to no improvement in validation loss.")
            break
    
    torch.save(model.state_dict(), f'crepe_{model_capacity}_final.pth')
    #wandb.finish()
    
if __name__ == "__main__":
    main()
