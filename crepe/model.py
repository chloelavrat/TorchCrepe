"""This file contains the Crepe model and its block."""

import torch
import os
import torchaudio
import torch.nn as nn

from crepe.utils import get_frame, activation_to_frequency


class ConvBlock(nn.Module):
    """
    Convolutional block model.

    Args:
        nn.Module (nn.Module): import torch.nn as nn
    """

    def __init__(self, out_channels, kernel_width, stride, in_channels):
        """
        Convolutional block with one or more convolutional layers.

        Args:
            out_channels (int): The number of output channels.
            kernel_width (int): The width of the convolutional kernel.
            stride (tuple, int): The stride for each dimension.
            in_channels (int): The number of input channels.

        Attributes:
            layer (nn.Sequential): A sequential container holding the block's layers.

        Methods:
            forward(x): Passes an input tensor `x` through the block.
        """
        super(ConvBlock, self).__init__()

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
        Pass an input tensor `x` through the network.

        Args:
            x (torch.Tensor): The input tensor to pass through the network.

        Returns:
            torch.Tensor: The output of the network.
        """
        return self.layer(x)


class Crepe(nn.Module):
    """
    Crepe model.

    Args:
        nn.Module (nn.Module): import torch.nn as nn
    """

    def __init__(self, model_capacity="full", device='cpu'):
        """
        CREPE model for pitch estimation.

        Args:
            model_capacity (str): The capacity of the network ('tiny', 'small', 'medium', 'large', or 'full').
            device (str): The device to run the model on ('cpu', 'gpu', 'mps').

        Attributes:
            model_capacity (str): The capacity of the network.
            convolutional_blocks (nn.Sequential): A sequential container holding the convolutional blocks.
            linear (nn.Linear): A linear layer for final mapping.

        Methods:
            forward(x): Passes an input tensor `x` through the network.
            get_activation(audio, sr, center=True, step_size=10, batch_size=128):
                Computes the activation stack for a given audio signal and sampling rate.
            predict(audio, sr, center=True, step_size=10, batch_size=128):
                Predicts pitch class labels from an input audio signal.

        Note:
            The model's capacity determines its size and complexity.
        """
        super(Crepe, self).__init__()

        # Define a multiplier for the network's capacity based on the selected model size
        self.model_capacity = model_capacity
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
            layers.append(ConvBlock(
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
        """
        Load the weights for a given model capacity.

        Args:
            model_capacity (str): The capacity of the network ('tiny', 'small', 'medium', 'large', or 'full').

        Note:
            The model's capacity determines its size and complexity.
        """
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "crepe-{}.pth".format(model_capacity)
        try:
            self.load_state_dict(torch.load(os.path.join(
                package_dir, filename), map_location=torch.device(self.device), weights_only=True))
        except:
            print(f"{filename} Not found.")

    def forward(self, x):
        """
        Pass an input tensor `x` through the network.

        Args:
            x (torch.Tensor): The input tensor to pass through the network.

        Returns:
            torch.Tensor: The output of the network.
        """
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

    def get_activation(self, audio, sr, center=True, step_size=10, batch_size=128):
        """
        Compute the activation stack for a given audio signal and sampling rate.

        Args:
            audio (torch.Tensor): The input audio tensor.
            sr (int): The sampling rate of the audio signal.
            center (bool): Whether to center the frames around each other. Defaults to True.
            step_size (int): The number of samples per frame. Defaults to 10.
            batch_size (int): The batch size for computing activations.

        Returns:
            torch.Tensor: The activation stack.
        """
        # resample to 16kHz if needed
        if sr != 16000:
            rs = torchaudio.transforms.Resample(sr, 16000)
            audio = rs(audio)

        # make mono if needed
        if len(audio.shape) == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                audio = audio.mean(dim=0)

        frames = get_frame(audio, step_size, center)
        activation_stack = []
        device = self.linear.weight.device

        for i in range(0, len(frames), batch_size):
            f = frames[i:min(i+batch_size, len(frames))]
            f = f.to(device)
            act = self.forward(f)
            activation_stack.append(act.cpu())
        activation = torch.cat(activation_stack, dim=0)

        return activation

    def predict(self, audio, sr, center=True, step_size=10, batch_size=128):
        """
        Predict pitch class labels from an input audio signal.

        Args:
            audio (torch.Tensor): The input audio tensor.
            sr (int): The sampling rate of the audio signal.
            center (bool): Whether to center the frames around each other. Defaults to True.
            step_size (int): The number of samples per frame. Defaults to 10.
            batch_size (int): The batch size for computing activations.

        Returns:
            tuple: A tuple containing the time, frequency, confidence, and activation stack.
        """
        activation = self.get_activation(
            audio, sr, batch_size=batch_size, step_size=step_size)
        frequency = activation_to_frequency(activation)
        confidence = activation.max(dim=1)[0]
        time = torch.arange(confidence.shape[0]) * step_size / 1000.0
        return time, frequency, confidence, activation
