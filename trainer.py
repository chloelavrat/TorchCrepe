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
from dataset import MIR1KDataset, Back10Dataset, NSynthDataset

import torch

def frequency_to_cents(frequencies, reference_frequency=10):
    # Avoid log of zero by setting minimum frequency to a small positive value
    min_frequency = 1e-6
    frequencies = torch.clamp(frequencies, min=min_frequency)
    
    # Convert frequency to cents
    cents = 1200 * torch.log2(frequencies / reference_frequency)
    
    return cents

def cents_to_activation(cents, num_bins=360, device='cpu'):
    """
    Converts cent values to activation maps.

    Args:
    - cents (torch.Tensor): Tensor of cent values.
    - num_bins (int): Number of bins for activation, default is 360.
    - device (str): Device to use for tensors, e.g., 'cpu' or 'mps'.

    Returns:
    - activation (torch.Tensor): Tensor of activation maps.
    """
    activation = torch.zeros((len(cents), num_bins), dtype=torch.float32, device=device)
    
    # Define the bin edges for the activation map
    bin_edges = torch.linspace(-1997.3794084376191, 7180, num_bins + 1, device=device)
    
    # Ensure the cent values are on the correct device
    cents = cents.to(device)
    
    for i, cent in enumerate(cents):
        # Find the bin index for the current cent value
        bin_index = torch.bucketize(cent, bin_edges, right=True) - 1
        bin_index = torch.clamp(bin_index, 0, num_bins - 1)
        
        # Set the corresponding bin to 1
        activation[i, bin_index] = 1.0
    
    return activation


def get_activation(labels, center=True, step_size=10, batch_size=128):

    if len(labels.shape) == 2:
        if labels.shape[0] == 1:
            labels = labels[0]
        else:
            labels = labels.mean(dim=0) # make mono    

    def get_frame(labels, step_size, center):
        if center:
            labels = nn.functional.pad(labels, pad=(512, 512))
        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        hop_length = int(16000 * step_size / 1000)
        n_frames = 1 + int((len(labels) - 1024) / hop_length)
        assert labels.dtype == torch.float32
        itemsize = 1 # float32 byte size
        frames = torch.as_strided(labels, size=(1024, n_frames), stride=(itemsize, hop_length * itemsize))
        frames = frames.transpose(0, 1).clone()

        frames -= (torch.mean(frames, axis=1).unsqueeze(-1))
        frames /= (torch.std(frames, axis=1).unsqueeze(-1))
        return frames    
    
    frames = get_frame(labels, step_size, center)
    
    activation_stack = []

    for i in range(0, len(frames), batch_size):
        f = frames[i:min(i+batch_size, len(frames))]
        f = F.interpolate(f.unsqueeze(0), size = 360, mode='linear', align_corners = False).squeeze(0)
        activation_stack.append(f.cpu())
    activation = torch.cat(activation_stack, dim=0)
    return activation

def train_epoch(model, dataloader, criterion, optimizer, sr, device):
    model.train()
    running_loss = 0.0
    for audio, labels in tqdm(dataloader):
        audio = audio.to(device)
        labels = labels.to(device)
        labels = frequency_to_cents(labels)
        labels = F.interpolate(labels.unsqueeze(0), size = audio.shape[-1], mode='linear', align_corners = False).squeeze(0)
        
        optimizer.zero_grad()
        model_activations = model.get_activation(audio, sr)
        label_activations = cents_to_activation(get_activation(labels))
        
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

            labels = frequency_to_cents(labels)
            labels = F.interpolate(labels.unsqueeze(0), size = audio.shape[-1], mode='linear', align_corners = False).squeeze(0)
            
            model_activations = model.get_activation(audio, sr)
            label_activations = cents_to_activation(get_activation(labels))
            
            loss = criterion(model_activations, label_activations)

            running_loss += loss.item() * audio.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    batch_size = 1
    num_epoch = 5000
    num_batches_per_epoch = 1
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
    
    for epoch in range(1, num_epoch):
        train_loss = 0.0
        for _ in range(num_batches_per_epoch):
            train_loss += train_epoch(model, train_loader, criterion, optimizer, sr, device)
        train_loss /= num_batches_per_epoch
        
        val_loss = validate_epoch(model, val_loader, criterion, sr, device)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
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
if __name__ == "__main__":
    main()
