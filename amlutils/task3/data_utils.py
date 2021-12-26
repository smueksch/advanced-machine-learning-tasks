import pickle
import gzip

import torch
from torch.utils.data import TensorDataset, DataLoader


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
