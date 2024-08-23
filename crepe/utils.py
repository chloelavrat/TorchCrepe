import torch
import torch.nn as nn

def get_frame(audio, step_size, center):
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
    Find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # The bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                torch.linspace(0, 1200 * torch.log2(torch.tensor(3951.066/10)), 360, dtype=salience.dtype, device=salience.device) + 1200 * torch.log2(torch.tensor(32.70/10)))

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
    
def activation_to_freq(activations):
    cents = to_local_average_cents(activations)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    frequency = torch.where(frequency < 32.71, torch.tensor(1e-7 ,device=frequency.device), frequency)
    return frequency
    