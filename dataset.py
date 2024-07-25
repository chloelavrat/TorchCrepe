import os
import torch
import torchaudio
import glob
import json
from torch.utils.data import Dataset, ConcatDataset

class MIR1KDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.labels = [f.replace('.wav', '.txt') for f in self.files]
        self.files = sorted(glob.glob(os.path.join(self.root_dir+"/Wavfile", f"*.wav")))
        self.labels = sorted(glob.glob(os.path.join(self.root_dir+"/PitchLabel", f"*.pv")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = os.path.abspath(self.files[idx])
        label_path = self.labels[idx]

        audio, sr = torchaudio.load(audio_path)
        
        with open(label_path, 'r') as f:
            labels = [float(line.strip()) for line in f.readlines()]

        labels = torch.tensor(labels)
        
        return audio[1,:], labels
    
class Back10Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_violin = sorted(glob.glob(os.path.join(self.root_dir, f"*/*violin.wav"), recursive=True))
        self.files_clarinet = sorted(glob.glob(os.path.join(self.root_dir, f"*/*clarinet.wav"), recursive=True))
        self.files_saxophone = sorted(glob.glob(os.path.join(self.root_dir, f"*/*bassoon.wav"), recursive=True))
        self.files_bassoon = sorted(glob.glob(os.path.join(self.root_dir, f"*/*bassoon.wav"), recursive=True))
        self.dataset_orga = {}

        idx = 0
        for path in self.files_violin:
            self.dataset_orga[idx]={}
            self.dataset_orga[idx]["type"]="violin"
            self.dataset_orga[idx]["number"]=1
            self.dataset_orga[idx]["audio_path"]=path
            self.dataset_orga[idx]["label_path"]=path.replace('-violin.wav', '.txt')
            idx += 1
        
        for path in self.files_clarinet:
            self.dataset_orga[idx]={}
            self.dataset_orga[idx]["type"]="clarinet"
            self.dataset_orga[idx]["number"]=2
            self.dataset_orga[idx]["audio_path"]=path
            self.dataset_orga[idx]["label_path"]=path.replace('-clarinet.wav', '.txt')
            idx += 1

        for path in self.files_saxophone:
            self.dataset_orga[idx]={}
            self.dataset_orga[idx]["type"]="saxophone"
            self.dataset_orga[idx]["number"]=3
            self.dataset_orga[idx]["audio_path"]=path
            self.dataset_orga[idx]["label_path"]=path.replace('-saxophone.wav', '.txt')
            idx += 1

        for path in self.files_bassoon:
            self.dataset_orga[idx]={}
            self.dataset_orga[idx]["type"]="bassoon"
            self.dataset_orga[idx]["number"]=4
            self.dataset_orga[idx]["audio_path"]=path
            self.dataset_orga[idx]["label_path"]=path.replace('-bassoon.wav', '.txt')
            idx += 1
        
        self.len = idx

        self.labels = sorted(glob.glob(os.path.join(self.root_dir, f"*/*.txt"), recursive=True))


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
        return self.len

    def __getitem__(self, idx):        
        instr_type = self.dataset_orga[idx]['number']
        audio_path = self.dataset_orga[idx]['audio_path']
        label_path = self.dataset_orga[idx]['label_path']

        audio, sr = torchaudio.load(audio_path)
        label = self._load_data(label_path, instr_type)
        label = torch.tensor(label)
        audio = torch.mean(audio, dim=0)
        
        return audio, label
    
class NSynthDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(self.root_dir, f"*/*/*.wav"), recursive=True))
        self.infos = sorted(glob.glob(os.path.join(self.root_dir, f"*/*.json"), recursive=True))
        
        self.data = {}
        for json_file in self.infos:
            with open(json_file, 'r') as f:
                self.data.update(json.load(f))
        
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        filename = [*self.data][idx]
        audio_path = [s for s in self.files if filename in s]

        audio, sr = torchaudio.load(audio_path[0])
        audio = torch.mean(audio, dim=0)

        midi_pitch = self.data[filename]["pitch"]
        pitch = 440 * 2**((midi_pitch - 69) / 12)
        pitch = torch.ones([audio.shape[0]//40]) * pitch

        return audio, pitch

    def test(self):
        print(self.data)
        
if __name__ == "__main__":
    
    mir_1k = MIR1KDataset(root_dir="./dataset/MIR-1K")
    print("** mir_1k")
    print(len(mir_1k))
    print("audio size: ", mir_1k[32][0].shape)
    print("label size: ", mir_1k[32][1].shape)
    
    back10 = Back10Dataset(root_dir="./dataset/bach10")
    print("** back10")
    print(len(back10))
    print("audio size: ", back10[1][0].shape)
    print("label size: ", back10[1][1].shape)
    
    nsynth = NSynthDataset(root_dir="./dataset/Nsynth-mixed")
    print("** nsynth")
    print(len(nsynth))
    print("audio size: ", nsynth[45][0].shape)
    print("label size: ", nsynth[45][1].shape)
    
    
    combined_dataset = ConcatDataset([mir_1k, back10, nsynth])
    print("** combined_dataset")
    print(len(combined_dataset))
    print("audio size: ", combined_dataset[45][0].shape)
    print("label size: ", combined_dataset[45][1].shape)
    
    