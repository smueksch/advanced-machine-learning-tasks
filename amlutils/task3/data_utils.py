import pickle
import gzip

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from typing import Dict

from .data_augmentation import generate_augmented_data, pad_to_multiple


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def convert_to_tensor(X, data_type=torch.float):
    X_tensor = torch.tensor(X, dtype=data_type)
    # Unsqueeze X tensor to have another dimension representing the channel, this
    # is needed for convolutions.
    X_tensor = torch.unsqueeze(X_tensor, 0)
    return X_tensor


def build_data_loader(X, y):
    X_tensor = convert_to_tensor(X)
    y_tensor = convert_to_tensor(y, torch.long)

    train_tensor = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset=train_tensor, batch_size=1, shuffle=True)


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict]):
        """Data must have keys 'frame', 'label' and 'box'."""
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        return self.data[index]


def build_augmented_dataset_loader(data, batch_size):
    dataset = AugmentedDataset(data)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
