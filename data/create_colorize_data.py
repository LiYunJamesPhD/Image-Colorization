import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path

from torch.utils.data import DataLoader
from .colorization_dataset import ColorDataset, TestColorDataset


def split_imges(dir_path, percent_val):

    # create train paths and val paths
    train_paths = []
    val_paths = []

    img_paths = glob.glob(dir_path)
    np.random.seed(123)
    img_num = len(img_paths)

    rand_idxs = np.random.permutation(img_num)
    train_num = int(img_num * percent_val)

    train_idxs = rand_idxs[:train_num]
    for idx in train_idxs:
        train_paths += [img_paths[idx]]

    if percent_val != 1.0:
        val_idxs = rand_idxs[train_num:]
        
        for idx in val_idxs:
            val_paths += [img_paths[idx]]

    return train_paths, val_paths

def make_test_color_dataloaders(batch_size=1, n_workers=0, pin_memory=True, **kwargs):
    """
    create dataset and dataloader for test images (grayscale images)
    """
    test_dataset = TestColorDataset(**kwargs)
    dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=n_workers, pin_memory=pin_memory)

    return dataloader

def make_color_dataloaders(batch_size=1, n_workers=0, pin_memory=True, **kwargs):
    """
    create dataset and dataloader
    """
    dataset = ColorDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=n_workers, pin_memory=pin_memory)

    return dataloader


