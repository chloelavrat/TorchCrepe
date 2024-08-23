import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
from crepe import CREPE
from utilsA*
from dataset import MIR1KDataset, Back10Dataset, NSynthDataset


def train_epoch(model, dataloader, criterion, optimizer, sr, device, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (audio, labels) in enumerate(tqdm(dataloader)):
        audio = audio.to(device)
        labels = labels.to(device)
        labels = F.interpolate(labels.unsqueeze(
            0), size=audio.shape[-1], mode='linear', align_corners=False).squeeze(0)
        labels = frequency_to_cents(labels)
        label_activations = get_activation_from_label(labels)

        with torch.amp.autocast(str(device)):
            model_activations = model.get_activation(audio, sr).to(device)
            loss = criterion(model_activations, label_activations)
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
            labels = F.interpolate(labels.unsqueeze(
                0), size=audio.shape[-1], mode='linear', align_corners=False).squeeze(0)
            labels = frequency_to_cents(labels)
            label_activations = get_activation_from_label(labels)

            model_activations = model.get_activation(audio, sr).to(device)

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

    device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    model = CREPE(model_capacity=model_capacity, device=device).to(device)

    mir_1k = MIR1KDataset(root_dir="./dataset/MIR-1K")
    back10 = Back10Dataset(root_dir="./dataset/Bach10")
    # nsynth = NSynthDataset(root_dir="./dataset/Nsynth-mixed")

    dataset = ConcatDataset([mir_1k, back10])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            num_workers=4, pin_memory=True)

    criterion = nn.nn.BCELoss()()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    scaler = torch.amp.GradScaler(device='cuda')

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    accumulation_steps = 4

    wandb.init(project="Crepe tiny", config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "MIR-1K",
        "epochs": num_epoch,
    })

    for epoch in range(1, num_epoch + 1):
        train_loss = 0.0
        for _ in range(num_batches_per_epoch):
            train_loss += train_epoch(model, train_loader, criterion,
                                      optimizer, sr, device, scaler, accumulation_steps)
        train_loss /= num_batches_per_epoch

        val_loss = validate_epoch(model, val_loader, criterion, sr, device)
        print(f'Epoch {epoch}, Train Loss: {
              train_loss:.4f}, Val Loss: {val_loss:.4f}')
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

        scheduler.step()

    torch.save(model.state_dict(), f'crepe_{model_capacity}_final.pth')
    wandb.finish()


if __name__ == "__main__":
    main()
