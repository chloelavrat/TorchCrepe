"""This file contains various dataloader for training and processing audio data and labels."""

from torch.utils.data import Dataset
import os
import torch
import torchaudio
import glob
import json
from torch.utils.data import Dataset, ConcatDataset
from concurrent.futures import ThreadPoolExecutor, as_completed


class MIR1KDataset(Dataset):
    """
    MIR-1K Dataset.

    Args:
        Dataset (Dataset): from torch.utils.data import Dataset
    """

    def __init__(self, root_dir):
        """
        Create an instance of the MIR-1K dataset.

        This class loads and prepares the data from the MIR-1K dataset,
        which consists of audio files with corresponding pitch labels.

        Attributes:
            root_dir (str): The root directory containing the MIR-1K dataset.

        Methods:
            __init__: Initializes the dataset by loading the audio files and pitch labels.
            __len__: Returns the total number of samples in the dataset.
            __getitem__: Loads a single sample from the dataset, consisting of an audio file
                and its corresponding pitch label.
        """
        self.root_dir = root_dir
        # self.labels = [f.replace('.wav', '.txt') for f in self.files]
        self.files = sorted(glob.glob(os.path.join(
            self.root_dir+"/Wavfile", f"*.wav")))
        self.labels = sorted(glob.glob(os.path.join(
            self.root_dir+"/PitchLabel", f"*.pv")))

    def __len__(self):
        """
        Retrieve the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio file and its corresponding pitch label,
                where the audio is a tensor representing the audio waveform, and the
                pitch label is a tensor containing the pitch values for each frame.

        Note:
            The returned audio tensor has shape (1, num_frames), where num_frames is the
            number of frames in the original audio file. The returned pitch label tensor has
            shape (num_frames,), containing one pitch value per frame.
        """
        audio_path = os.path.abspath(self.files[idx])
        label_path = self.labels[idx]

        audio, sr = torchaudio.load(audio_path)

        with open(label_path, 'r') as f:
            labels = [float(line.strip()) for line in f.readlines()]

        labels = torch.tensor(labels)

        return audio[1, :], labels


class Back10Dataset(Dataset):
    """
    Bach10 Dataset.

    Args:
        Dataset (Dataset): from torch.utils.data import Dataset
    """

    def __init__(self, root_dir):
        """
        Create an instance of the Bach10 dataset.

        This class loads and prepares the data from the Bach10 dataset,
        which consists of audio files with corresponding pitch labels.

        Attributes:
            root_dir (str): The root directory containing the Bach10 dataset.

        Methods:
            __init__: Initializes the dataset by loading the audio files and pitch labels.
            __len__: Returns the total number of samples in the dataset.
            __getitem__: Loads a single sample from the dataset, consisting of an audio file
                and its corresponding pitch label.
        """
        self.root_dir = root_dir
        self.files_violin = sorted(glob.glob(os.path.join(
            self.root_dir, f"*/*violin.wav"), recursive=True))
        self.files_clarinet = sorted(glob.glob(os.path.join(
            self.root_dir, f"*/*clarinet.wav"), recursive=True))
        self.files_saxophone = sorted(glob.glob(os.path.join(
            self.root_dir, f"*/*saxophone.wav"), recursive=True))
        self.files_bassoon = sorted(glob.glob(os.path.join(
            self.root_dir, f"*/*bassoon.wav"), recursive=True))
        self.dataset_orga = {}

        idx = 0
        for path in self.files_violin:
            self.dataset_orga[idx] = {}
            self.dataset_orga[idx]["type"] = "violin"
            self.dataset_orga[idx]["number"] = 1
            self.dataset_orga[idx]["audio_path"] = path
            self.dataset_orga[idx]["label_path"] = path.replace(
                '-violin.wav', '.txt')
            idx += 1

        for path in self.files_clarinet:
            self.dataset_orga[idx] = {}
            self.dataset_orga[idx]["type"] = "clarinet"
            self.dataset_orga[idx]["number"] = 2
            self.dataset_orga[idx]["audio_path"] = path
            self.dataset_orga[idx]["label_path"] = path.replace(
                '-clarinet.wav', '.txt')
            idx += 1

        for path in self.files_saxophone:
            self.dataset_orga[idx] = {}
            self.dataset_orga[idx]["type"] = "saxophone"
            self.dataset_orga[idx]["number"] = 3
            self.dataset_orga[idx]["audio_path"] = path
            self.dataset_orga[idx]["label_path"] = path.replace(
                '-saxophone.wav', '.txt')
            idx += 1

        for path in self.files_bassoon:
            self.dataset_orga[idx] = {}
            self.dataset_orga[idx]["type"] = "bassoon"
            self.dataset_orga[idx]["number"] = 4
            self.dataset_orga[idx]["audio_path"] = path
            self.dataset_orga[idx]["label_path"] = path.replace(
                '-bassoon.wav', '.txt')
            idx += 1

        self.len = idx

        self.labels = sorted(glob.glob(os.path.join(
            self.root_dir, f"*/*.txt"), recursive=True))

    def _load_data(self, file_path, instrument_number):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Ignore empty lines
                    # Parse the line
                    parts = line.strip().split()
                    time_audio = int(parts[0])
                    time_midi = int(parts[1])
                    midi_pitch = int(parts[2])
                    channel = int(parts[3])

                    if channel == instrument_number:
                        # Convert MIDI pitch to fundamental frequency
                        frequency = 440 * 2**((midi_pitch - 69) / 12)
                        data.append(frequency)
        return data

    def __len__(self):
        """
        Retrieve the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio file and its corresponding pitch label,
                where the audio is a tensor representing the audio waveform, and the
                pitch label is a tensor containing the pitch values for each frame.
                (audio, label)

        Note:
            The returned audio tensor has shape (1, num_frames), where num_frames is the
            number of frames in the original audio file. The returned pitch label tensor has
            shape (num_frames,), containing one pitch value per frame.
        """
        instr_type = self.dataset_orga[idx]['number']
        audio_path = self.dataset_orga[idx]['audio_path']
        label_path = self.dataset_orga[idx]['label_path']

        audio, sr = torchaudio.load(audio_path)
        label = self._load_data(label_path, instr_type)
        label = torch.tensor(label)
        audio = torch.mean(audio, dim=0)

        return audio, label


class NSynthDataset(Dataset):
    """
    Nsynth Dataset.

    Args:
        Dataset (Dataset): from torch.utils.data import Dataset
    """

    def __init__(self, root_dir, n_samples=1):
        """
        Create an instance of the NSynth dataset.

        This class loads and prepares the data from the NSynth dataset,
        which consists of audio files with corresponding pitch labels.

        Attributes:
            root_dir (str): The root directory containing the NSynth dataset.

        Methods:
            __init__: Initializes the dataset by loading the audio files and pitch labels.
            __len__: Returns the total number of samples in the dataset.
            __getitem__: Loads a single sample from the dataset, consisting of an audio file
                and its corresponding pitch label.
        """
        self.root_dir = root_dir
        self.n_samples = n_samples

        # Load file paths
        self.files = sorted(glob.glob(os.path.join(
            root_dir, "*/*/*.wav"), recursive=True))
        self.infos = sorted(glob.glob(os.path.join(
            root_dir, "*/*.json"), recursive=True))

        # Precompute a mapping from filenames to their paths
        self.filename_to_path = {os.path.splitext(os.path.basename(f))[
            0]: f for f in self.files}

        # Load all metadata
        self.data = self._load_metadata()

        # Create a mapping from filenames to their data for fast access
        self.file_to_data = {filename: self.data.get(
            filename, {}) for filename in self.data}

    def _load_metadata(self):
        data = {}
        for json_file in self.infos:
            with open(json_file, 'r') as f:
                data.update(json.load(f))
        return data

    def __len__(self):
        """
        Retrieve the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.files) // self.n_samples

    def _load_audio_and_pitch(self, filename):
        audio_path = self.filename_to_path[filename]
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0)  # Convert to mono

        # Retrieve MIDI pitch and calculate frequency
        midi_pitch = self.file_to_data.get(filename, {}).get(
            "pitch", 69)  # Default to 69 if not found
        pitch = 440 * 2 ** ((midi_pitch - 69) / 12)
        pitch = torch.ones([audio.shape[0] // 40]) * pitch

        return audio, pitch

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio file and its corresponding pitch label,
                where the audio is a tensor representing the audio waveform, and the
                pitch label is a tensor containing the pitch values for each frame.
                (audio, label)

        Note:
            The returned audio tensor has shape (1, num_frames), where num_frames is the
            number of frames in the original audio file. The returned pitch label tensor has
            shape (num_frames,), containing one pitch value per frame.
        """
        start_idx = idx * self.n_samples
        end_idx = start_idx + self.n_samples

        # Ensure we don't go out of bounds
        if end_idx > len(self.files):
            raise IndexError(
                "Index out of bounds for the requested group of samples.")

        audio_list = []
        pitch_list = []

        filenames = list(self.data.keys())[start_idx:end_idx]

        with ThreadPoolExecutor() as executor:
            future_to_filename = {executor.submit(
                self._load_audio_and_pitch, filename): filename for filename in filenames}
            for future in as_completed(future_to_filename):
                audio, pitch = future.result()
                audio_list.append(audio)
                pitch_list.append(pitch)

        audio = torch.cat(audio_list)
        pitch = torch.cat(pitch_list)

        return audio, pitch
