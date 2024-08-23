import torch.nn.functional as F
import torch.nn as nn
import torch


def activation_to_frequency(activation):
    """
    Convert activation tensor to frequency tensor using local average cents.

    Args:
        activation (torch.Tensor): The activation tensor with shape (batch_size, num_bins).

    Returns:
        torch.Tensor: The frequency tensor.
    """
    def compute_local_average_cents(salience, center=None):
        """
        Compute the weighted average of cents around the center bin.

        Args:
            salience (torch.Tensor): The salience tensor (1D or 2D).
            center (int, optional): The bin index to center around. Defaults to the index of the maximum value in salience.

        Returns:
            torch.Tensor: The average cents value.
        """
        if not hasattr(compute_local_average_cents, 'cents_mapping'):
            # Bin number-to-cents mapping
            cents_mapping = torch.linspace(
                0,
                1200 * torch.log2(torch.tensor(3951.066 / 10)),
                360,
                dtype=salience.dtype,
                device=salience.device
            ) + 1200 * torch.log2(torch.tensor(32.70 / 10))
            compute_local_average_cents.cents_mapping = cents_mapping

        if salience.ndim == 1:
            if center is None:
                center = int(torch.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience_segment = salience[start:end]
            cents_segment = compute_local_average_cents.cents_mapping[start:end]
            weighted_sum = torch.sum(salience_segment * cents_segment)
            total_weight = torch.sum(salience_segment)
            return weighted_sum / total_weight
        elif salience.ndim == 2:
            return torch.stack([compute_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])
        else:
            raise ValueError("Input tensor must be either 1D or 2D.")

    cents = compute_local_average_cents(activation)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    return frequency


def frequency_to_activation(frequencies, center=True, step_size_ms=10, batch_size=128):
    """
    Converts frequency data into an activation tensor.

    This function processes 1D frequency data or 2D frequency data (in which case it averages across channels to create a mono signal).
    It extracts frames from the frequency data, computes mean frequencies for each frame, maps these frequencies to bins,
    and generates an activation tensor based on these bins.

    Parameters:
    - frequencies (torch.Tensor): A 1D or 2D tensor of frequency data. If 2D, the tensor is averaged across channels.
    - center (bool, optional): If True, pads the frequency data with zeros on both sides to ensure frames are centered. Default is True.
    - step_size_ms (int, optional): The step size in milliseconds for frame extraction. Default is 10 ms.
    - batch_size (int, optional): The batch size is currently unused in this function but included for potential future use. Default is 128.

    Returns:
    - torch.Tensor: A 2D tensor of shape (num_frames, 360) representing the activations. Each row corresponds to a frame and each column to a bin.
    """

    # Convert multi-channel input to mono if necessary
    if frequencies.dim() == 2:
        if frequencies.size(0) == 1:
            frequencies = frequencies.squeeze(0)
        else:
            frequencies = frequencies.mean(dim=0)

    device = frequencies.device

    def extract_frames(frequencies, step_size_ms, center):
        """
        Extracts overlapping frames from the frequency data.

        Parameters:
        - frequencies (torch.Tensor): The frequency data from which to extract frames.
        - step_size_ms (int): The step size in milliseconds for frame extraction.
        - center (bool): If True, pads the frequency data to center the frames.

        Returns:
        - torch.Tensor: A 2D tensor of shape (num_frames, frame_length) containing the extracted frames.
        """
        if center:
            frequencies = F.pad(frequencies, pad=(512, 512))
        # Calculate hop length in samples
        sample_rate = 16000
        hop_length = int(sample_rate * step_size_ms / 1000)
        # Calculate number of frames
        frame_length = 1024
        num_frames = 1 + (frequencies.size(0) - frame_length) // hop_length
        # Extract frames
        frames = torch.as_strided(
            frequencies, size=(
                frame_length, num_frames), stride=(1, hop_length)
        )
        frames = frames.transpose(0, 1).clone()
        return frames

    def cents_to_bin_indices(cents_values, num_bins=360, dtype=torch.float32):
        """
        Maps cent values to bin indices based on predefined bin edges.

        Parameters:
        - cents_values (torch.Tensor): The cent values to map to bins.
        - num_bins (int): The number of bins to map to.
        - dtype (torch.dtype): The data type for the bin edges.

        Returns:
        - torch.Tensor: A tensor of bin indices corresponding to the cent values.
        """
        # Compute bin edges in cents
        min_cents = 1200 * \
            torch.log2(torch.tensor(32.70 / 10, device=cents_values.device))
        max_cents = 1200 * \
            torch.log2(torch.tensor(3951.066 / 10,
                       device=cents_values.device)) + min_cents
        bin_edges = torch.linspace(
            min_cents, max_cents, num_bins, dtype=dtype, device=cents_values.device)
        # Determine bin indices
        bin_indices = torch.bucketize(cents_values, bin_edges, right=True)
        bin_indices = torch.clamp(bin_indices, min=0, max=num_bins - 1)
        return bin_indices

    # Extract frames from the frequency data
    frames = extract_frames(frequencies, step_size_ms, center)
    frames = frames.float()

    # Compute the mean frequency across frames
    mean_frequencies = frames.mean(dim=-1)
    bin_indices = cents_to_bin_indices(mean_frequencies).to(device)

    # Create activation tensor
    activation = torch.zeros(frames.size(0), 360, device=device)
    activation.scatter_(1, bin_indices.unsqueeze(1), 1)

    return activation
