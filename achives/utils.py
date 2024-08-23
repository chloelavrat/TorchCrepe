import torch.nn as nn
import torch
import os

def frequency_to_cents(frequencies, reference_frequency=10):
    # Avoid log of zero by setting minimum frequency to a small positive value
    min_frequency = 1e-6
    frequencies = torch.clamp(frequencies, min=min_frequency)
    
    # Convert frequency to cents
    cents = 1200 * torch.log2(frequencies / reference_frequency)
    
    return cents

def cent_to_bin_mapping(cents_values, dtype=torch.float32):
    # Define the mapping parameters
    num_bins = 360
    min_cents = 1200 * torch.log2(torch.tensor(32.70/10))
    max_cents = 1200 * torch.log2(torch.tensor(3951.066/10)) + min_cents
    bin_edges = torch.linspace(min_cents, max_cents, num_bins, dtype=dtype, device=cents_values.device)
    
    # Compute the bin index for each cent value
    # Find the closest bin edge for each cent value
    bin_indices = torch.bucketize(cents_values, bin_edges, right=True)
    
    # Ensure bin indices are within bounds
    bin_indices = torch.clamp(bin_indices, min=0, max=num_bins-1)
    
    return bin_indices

def get_activation_from_label(labels, center=True, step_size=10, batch_size=128):
    """     
    labels : (N,) or (C, N)
    """
    
    if len(labels.shape) == 2:
        if labels.shape[0] == 1:
            labels = labels[0]
        else:
            labels = labels.mean(dim=0) # make mono

    def get_frame(labels, step_size, center):
        if center:
            labels = nn.functional.pad(labels, pad=(512, 512))
        # make 1024-sample frames of the labels with hop length of 10 milliseconds
        hop_length = int(16000 * step_size / 1000)
        n_frames = 1 + (len(labels) - 1024) // hop_length
        frames = torch.as_strided(labels, size=(1024, n_frames), stride=(1, hop_length))
        frames = frames.transpose(0, 1).clone()
    
        return frames
        
    frames = get_frame(labels, step_size, center)
    activation_stack = []
    
    for i in range(0, len(frames), batch_size):
        f = frames[i:min(i+batch_size, len(frames))]
        act = torch.zeros([f.shape[0],360])
        mean_f = f.mean(dim=-1)
        bins = cent_to_bin_mapping(mean_f)
        for idx, a in enumerate(act):
            act[idx,bins[idx]]=1

        activation_stack.append(act.cpu())
    activation = torch.cat(activation_stack, dim=0)
    return activation

def download_weights(model_capacity):
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    weight_file = 'crepe-{}.pth'.format(model_capacity)
    base_url = 'https://github.com/sweetcocoa/crepe-pytorch/raw/models/'

    # In all other cases, decompress the weights file if necessary
    package_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(package_dir, weight_file)
    if not os.path.isfile(weight_path):
        print('Downloading weight file {} from {} ...'.format(weight_path, base_url + weight_file))
        urlretrieve(base_url + weight_file, weight_path)

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

    raise Exception("Label should be either 1D or 2D Tensor")

def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch continuity.

    Note: This is NOT implemented with PyTorch.
    """
    from hmmlearn import hmm
    import numpy as np

    # Uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # Transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # Emission probability = fixed probability for self, evenly distribute the others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones((360, 360)) * ((1 - self_emission) / 360))

    # Fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(n_components=360)
    model.startprob_ = starting
    model.transmat_ = transition
    model.emissionprob_ = emission

    # Find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), lengths=[len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])

def to_freq(activation, viterbi=False):
    if viterbi:
        cents = to_viterbi_cents(activation.detach().cpu().numpy())
        cents = torch.tensor(cents, dtype=torch.float32, device=activation.device)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    frequency = torch.where(frequency < 32.71, torch.tensor(1e-7 ,device=frequency.device), frequency)
    return frequency
