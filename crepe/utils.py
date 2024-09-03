"""This file contains various functions for processing and converting audio data and labels."""

import torch
import torch.nn as nn
import subprocess
import torchaudio
import os


def get_frame(audio, step_size, center):
    """
    Extract audio frames from a given audio signal.

    Args:
        audio (Tensor): The input audio signal.
        step_size (float): The time step size in milliseconds. Audio will be divided into
        1024-sample frames with this hop length.
        center (bool): If True, pads the audio to have equal number of samples on both sides.

    Returns:
        Tensor: A tensor containing the extracted audio frames, standardized to have zero mean and unit standard deviation.
    """
    if center:
        audio = nn.functional.pad(audio, pad=(512, 512))
    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(16000 * step_size / 1000)
    n_frames = 1 + (len(audio) - 1024) // hop_length
    frames = torch.as_strided(audio, size=(
        1024, n_frames), stride=(1, hop_length))
    frames = frames.transpose(0, 1).clone()

    mean = torch.mean(frames, dim=1, keepdim=True)
    # Adding epsilon to prevent division by zero
    std = torch.std(frames, dim=1, keepdim=True) + 1e-8

    frames -= mean
    frames /= std
    return frames


def to_local_average_cents(salience, center=None):
    """
    Compute the weighted average cents near the argmax bin of a salience vector.

    Args:
        salience (Tensor): A 1D or 2D tensor representing the salience values.
        center (int, optional): The index around which to compute the weighted average. Defaults to None.

    Returns:
        Tensor: The weighted average cents near the argmax bin.

    Notes:
        This function assumes that the input salience values are normalized such that their sum equals 1.
    """
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # The bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
            torch.linspace(0,
                           1200 * torch.log2(torch.tensor(3951.066/10)),
                           360, dtype=salience.dtype,
                           device=salience.device) + 1200 * torch.log2(torch.tensor(32.70/10)))

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience_segment = salience[start:end]
        mapping_segment = to_local_average_cents.cents_mapping[start:end]
        product_sum = torch.sum(salience_segment * mapping_segment)
        weight_sum = torch.sum(salience_segment)
        return product_sum / weight_sum
    elif salience.ndim == 2:
        return torch.stack([to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])


def activation_to_frequency(activations):
    """
    Convert activations to a corresponding frequency value.

    Args:
        activations (tensor): The input activations to convert.

    Returns:
        tensor: A tensor representing the frequency values.
    """
    cents = to_local_average_cents(activations)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    frequency = torch.where(frequency < 32.71, torch.tensor(
        1e-7, device=frequency.device), frequency)
    return frequency


def frequency_to_activation(frequencies, num_bins=360):
    """
    Convert a tensor of frequencies to a binary activation map.

    Args:
        frequencies (torch.Tensor): The input frequencies.
        num_bins (int, optional): The number of bins in the activation map. Defaults to 360.

    Returns:
        torch.Tensor: A binary activation map where each row corresponds to the frequency in the corresponding
        row of `frequencies`.
    """
    # Convert frequency to cents
    cents = 1200 * torch.log2(frequencies / 10)

    # Create the cents-to-bin mapping if it doesn't already exist
    if not hasattr(frequency_to_activation, 'cents_mapping'):
        frequency_to_activation.cents_mapping = (
            torch.linspace(0,
                           1200 * torch.log2(torch.tensor(3951.066/10)),
                           num_bins, dtype=frequencies.dtype,
                           device=frequencies.device) + 1200 * torch.log2(torch.tensor(32.70/10)))

    # Initialize activation map with zeros; expects batch input for frequencies
    activations = torch.zeros(
        frequencies.shape[0], num_bins, dtype=frequencies.dtype, device=frequencies.device)

    # Find the closest bin to the calculated cents value for each frequency in the batch
    for i in range(frequencies.shape[0]):
        closest_bin = torch.argmin(
            torch.abs(frequency_to_activation.cents_mapping - cents[i]))
        activations[i, closest_bin] = 1.0

    return activations


def load_test_file(filename: str, mono: bool = False, normalize: bool = False):
    """
    Load a test audio file from a remote server into a PyTorch Audio tensor.

    Args:
        filename (str): The name of the audio file to download.
        mono (bool, optional): If True, load the audio in monaural format. Defaults to False.
        normalize (bool, optional): If True, normalize the audio signal. Defaults to False.

    Returns:
        Tuple[Tensor, int]: A tuple containing the loaded audio tensor and its sample rate.
            If an error occurs during download or loading, returns (None, None).
    """
    # Construct the URL for the file on your server
    url = f"https://openfileserver.chloelavrat.com/testfiles/audio/{filename}"

    # Create a temporary directory to store the downloaded files
    temp_dir = "/tmp"  # You can change this to any other directory you like

    # Construct the full path where the downloaded file will be saved
    filepath = os.path.join(temp_dir, filename)

    try:
        # Use subprocess to run wget and download the file
        subprocess.check_call(
            ["wget", "-P", temp_dir, url, "--no-check-certificate", "-q"])

        # Load the audio file into a PyTorch Audio tensor
        audio, sr = torchaudio.load(filepath, normalize=normalize)

        return audio, sr

    except subprocess.CalledProcessError as e:
        print(f"Failed to download {filename}: {e}")
        return None
