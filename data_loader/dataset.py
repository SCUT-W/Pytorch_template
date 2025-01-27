import torch
from numpy import dtype
from torch.utils.data import Dataset
import numpy as np
import numpy as np
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, dataset_name, is_train=True):
        if is_train:
            self.spectrogram_path = f'data/{dataset_name}/train_data/log_mel_spec.npy'
            self.label_a_path = f'data/{dataset_name}/train_data/label_a.npy'
            self.label_v_path = f'data/{dataset_name}/train_data/label_v.npy'
        else:
            self.spectrogram_path = f'data/{dataset_name}/test_data/log_mel_spec.npy'
            self.label_a_path = f'data/{dataset_name}/test_data/label_a.npy'
            self.label_v_path = f'data/{dataset_name}/test_data/label_v.npy'

        self.spectrograms = torch.from_numpy(np.load(self.spectrogram_path)).float()
        self.is_train = is_train
        self.label_a = torch.from_numpy(np.load(self.label_a_path)).float()
        self.label_v = torch.from_numpy(np.load(self.label_v_path)).float()

    def __len__(self):
        return len(self.label_a)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = torch.stack([self.label_a[idx], self.label_v[idx]],dim=-1)
        return spectrogram, label
