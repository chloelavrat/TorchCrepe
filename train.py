import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from crepe.utils import frequency_to_activation
from crepe.model import Crepe
# import wandb


def epoch_step(model, audio, labels, sr, device):
    # send tensor to device
    audio = audio.to(device)
    labels = labels.to(device)

    # compute label activations
    labels_activations = frequency_to_activation(labels[0, :])

    # compute model activations
    model_activations = model.get_activation(audio, sr).to(device)

    labels_activations = F.interpolate(labels_activations.unsqueeze(0).unsqueeze(0), size=(
        model_activations.shape[0], labels_activations.shape[1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    # return activations
    return labels_activations, model_activations


def train_epoch(model, optimizer, dataloader, sr, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    criterion = nn.BCELoss()  # Initialize the loss function

    for i, (audio, labels) in enumerate(tqdm(dataloader)):
        audio, labels = audio.to(device), labels.to(
            device)  # Move data to the correct device

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        labels_activations, model_activations = epoch_step(
            model=model,
            audio=audio,
            labels=labels,
            sr=sr,
            device=device
        )

        # Compute the loss
        loss = criterion(model_activations, labels_activations)
        running_loss += loss.item() * audio.size(0)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, sr, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    criterion = nn.BCELoss()  # Initialize the loss function

    with torch.no_grad():  # Disable gradient calculation
        for i, (audio, labels) in enumerate(tqdm(dataloader)):
            audio, labels = audio.to(device), labels.to(
                device)  # Move data to the correct device

            # Forward pass
            labels_activations, model_activations = epoch_step(
                model=model,
                audio=audio,
                labels=labels,
                sr=sr,
                device=device
            )

            # Compute the loss
            loss = criterion(model_activations, labels_activations)
            running_loss += loss.item() * audio.size(0)

    return running_loss / len(dataloader.dataset)


if __name__ == "__main__":
    from crepe.dataset import MIR1KDataset, Back10Dataset, NSynthDataset
    from torch.utils.data import DataLoader, random_split, ConcatDataset

    model_capacity = 'small'
    learning_rate = 0.0002
    num_epoch = 50000
    num_batches_per_epoch = 8
    sr = 16000
    max_epochs_without_improvement = 32

    device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = Crepe(model_capacity=model_capacity).to(device)

    # dataset
    mir_1k = MIR1KDataset(root_dir="./dataset/MIR-1K")
    back10 = Back10Dataset(root_dir="./dataset/Bach10")
#    nsynth = NSynthDataset(root_dir="./dataset/Nsynth-mixed", n_samples=30)
    dataset = ConcatDataset([back10, mir_1k])

    # set train, validation dataset sizes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # wandb init
    # wandb.init(project="Crepe tiny", config={
    #     "learning_rate": learning_rate,
    #     "architecture": "CNN",
    #     "dataset": "MIR-1K + bach10",
    #     "epochs": num_epoch,
    # })

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    accumulation_steps = 4

    for epoch in range(1, num_epoch + 1):
        train_loss = 0.0
        for _ in range(num_batches_per_epoch):
            tmp_loss = train_epoch(
                model,
                optimizer,
                train_loader,
                sr,
                device
            )
            train_loss += tmp_loss
            # wandb.log({"train_loss": tmp_loss})
        # compute train loss
        train_loss /= num_batches_per_epoch

        # validation step
        val_loss = validate_epoch(
            model,
            val_loader,
            sr,
            device
        )

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # wandb.log({"val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f'crepe/crepe_{model_capacity}_best.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print("Stopping early due to no improvement in validation loss.")
            break

    # save model
    torch.save(model.state_dict(), f'crepe/crepe_{model_capacity}_final.pth')
    # wandb.finish()
