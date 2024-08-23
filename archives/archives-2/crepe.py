import os
import torch
import torchaudio
import torch.nn as nn
from utils import activation_to_frequency


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block that includes padding, convolution, activation,
    batch normalization, max pooling, and dropout, all organized within an
    nn.Sequential container.

    Parameters:
    -----------
    out_channels : int
        Number of output channels for the convolutional layer.

    kernel_width : int
        Width of the convolutional kernel (height dimension).

    stride : int
        Stride for the convolutional layer.

    in_channels : int
        Number of input channels for the convolutional layer.
    """

    def __init__(self, out_channels, kernel_width, stride, in_channels):
        super(ConvolutionalBlock, self).__init__()

        # Calculate padding for the height dimension (kernel width)
        pad_top = (kernel_width - 1) // 2
        pad_bottom = (kernel_width - 1) - pad_top

        # Define the block using nn.Sequential
        self.layer = nn.Sequential(
            # Add padding to the input
            nn.ZeroPad2d((0, 0, pad_top, pad_bottom)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_width, 1),
                stride=stride
            ),  # Apply 2D convolution
            nn.ReLU(),  # Apply ReLU activation
            nn.BatchNorm2d(out_channels),  # Apply batch normalization
            nn.MaxPool2d(kernel_size=(2, 1)),  # Apply max pooling
            nn.Dropout(p=0.25)  # Apply dropout for regularization
        )

    def forward(self, x):
        """
        Defines the forward pass through the convolutional block.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
        --------
        torch.Tensor
            Output tensor after passing through the convolutional block.
        """
        return self.layer(x)


class CREPE(nn.Module):
    """
    CREPE model for pitch detection using convolutional neural networks (CNNs).

    This model can be configured with different capacities, determining the
    size and complexity of the network.

    Parameters:
    -----------
    model_capacity : str, optional (default="full")
        The capacity of the model. Can be one of 'tiny', 'small', 'medium', 'large', 'full'.
        This determines the number of filters in each convolutional layer.
    """

    def __init__(self, model_capacity="full", device='cpu'):
        super(CREPE, self).__init__()

        # Define a multiplier for the network's capacity based on the selected model size
        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        # Define the number of filters for each layer, scaled by the capacity multiplier
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        # Include the input channel size as the first element
        filters = [1] + filters

        # Define the kernel widths and strides for each layer
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        # Create a list of layers for the Sequential container
        layers = []
        for i in range(len(filters) - 1):
            layers.append(ConvolutionalBlock(
                out_channels=filters[i + 1],
                kernel_width=widths[i],
                stride=strides[i],
                in_channels=filters[i]
            ))

        # Add all layers to the Sequential container
        self.convolutional_blocks = nn.Sequential(*layers)

        # Define the final linear layer to map the output to 360 classes (e.g., pitch classes)
        self.linear = nn.Linear(64 * capacity_multiplier, 360)

        # load model
        self.device = device
        self.load_weight(model_capacity)

        # Set the model to evaluation mode by default
        self.eval()

    def load_weight(self, model_capacity):
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "crepe-{}.pth".format(model_capacity)
        try:
            self.load_state_dict(torch.load(os.path.join(
                package_dir, filename), map_location=torch.device(self.device), weights_only=True))
        except:
            print(f"{filename} Not found.")

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, sample_length).

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, 360) containing the model's predictions.
        """
        # Reshape input to match the expected shape for convolutional layers
        x = x.view(x.shape[0], 1, -1, 1)

        # Pass the input through each convolutional block sequentially
        x = self.convolutional_blocks(x)

        # Reorder dimensions and flatten before passing to the linear layer
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)

        # Apply the final linear layer and sigmoid activation
        x = self.linear(x)
        x = torch.sigmoid(x)

        return x

    def _extract_frames(self, audio, step_ms, center, target_sample_rate):
        """
        Extract frames from the audio signal.

        Parameters:
        - audio (Tensor): The audio signal.
        - step_ms (int): Step size in milliseconds.
        - center (bool): Whether to pad the audio for centering.

        Returns:
        - Tensor: Normalized frames.
        """
        frame_length = 1024
        if center:
            padding = 512
            audio = nn.functional.pad(audio, pad=(padding, padding))

        hop_length = int(target_sample_rate * step_ms / 1000)
        num_frames = 1 + (len(audio) - frame_length) // hop_length

        frames = torch.as_strided(audio, size=(
            frame_length, num_frames), stride=(1, hop_length))
        frames = frames.transpose(0, 1).clone()

        # Normalize frames
        mean = torch.mean(frames, dim=1, keepdim=True)
        std = torch.std(frames, dim=1, keepdim=True) + 1e-8
        frames = (frames - mean) / std

        return frames

    def compute_activation(self, audio_signal, sample_rate, center_audio=True, frame_step_ms=10, batch_size=128):
        """
        Compute the activation of audio signal by processing it into frames and passing them through a model.

        Parameters:
        - audio_signal (Tensor): Input audio signal with shape (N,) or (C, N).
        - sample_rate (int): Sample rate of the input audio signal.
        - center_audio (bool, optional): Whether to center the audio by padding. Default is True.
        - frame_step_ms (int, optional): Step size in milliseconds for frame extraction. Default is 10 ms.
        - batch_size (int, optional): Batch size for processing frames through the model. Default is 128.

        Returns:
        - Tensor: Activation output of the audio signal.
        """

        # Resample audio if sample rate is not 16000 Hz
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, target_sample_rate)
            audio_signal = resampler(audio_signal)

        # Convert to mono if audio has more than one channel
        if len(audio_signal.shape) == 2:
            if audio_signal.shape[0] == 1:
                audio_signal = audio_signal[0]
            else:
                audio_signal = audio_signal.mean(dim=0)

        # Extract frames and compute activations
        frames = self._extract_frames(
            audio_signal, frame_step_ms, center_audio, target_sample_rate)
        activation_list = []

        # Ensure that the model is on the correct device
        device = self.linear.weight.device

        # Process frames in batches
        for start_idx in range(0, len(frames), batch_size):
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx].to(device)
            batch_activation = self.forward(batch_frames)
            activation_list.append(batch_activation.cpu())

        # Concatenate all activations
        activation_output = torch.cat(activation_list, dim=0)

        return activation_output

    def predict(self, audio, sr, frame_step_ms=10, batch_size=128):
        activation = self.compute_activation(
            audio, sr, batch_size=batch_size, frame_step_ms=frame_step_ms)
        frequency = activation_to_frequency(activation)
        confidence = activation.max(dim=1)[0]
        time = torch.arange(confidence.shape[0]) * frame_step_ms / 1000.0
        return time, frequency, confidence, activation


if __name__ == "__main__":

    crepe = CREPE(model_capacity='tiny')

    zeros = torch.zeros([160000])

    print(crepe.predict(zeros, sr=16000)[1].shape)
