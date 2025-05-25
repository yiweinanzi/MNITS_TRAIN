import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import struct

def read_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols).astype(np.float32) / 255.0
    return torch.tensor(data)

def read_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.tensor(labels, dtype=torch.long)

def load_mnist_from_local(root='./mnist_data'):
    files = {
        'train_images': 'train-images-idx3-ubyte',
        'train_labels': 'train-labels-idx1-ubyte',
        'test_images': 't10k-images-idx3-ubyte',
        'test_labels': 't10k-labels-idx1-ubyte'
    }
    paths = {k: os.path.join(root, v) for k, v in files.items()}
    train_images = read_idx_images(paths['train_images'])
    train_labels = read_idx_labels(paths['train_labels'])
    test_images = read_idx_images(paths['test_images'])
    test_labels = read_idx_labels(paths['test_labels'])

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    return train_dataset, test_dataset