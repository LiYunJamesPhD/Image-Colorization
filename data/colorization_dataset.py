from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class TestColorDataset(Dataset):

    def __init__(self, input_path, resize=256):
        """
        Input_path: a directory path to grayscale images
        resize: image resolution for image resizing
        """
        self.input_paths = input_path
        self.resize = resize

        self.transforms = transforms.Resize((self.resize, self.resize),  Image.BICUBIC)

    def __getitem__(self, idx):
        """Read grayscale images and normalize them"""
        file_name = self.input_paths[idx].split('/')[-1]
        img = Image.open(self.input_paths[idx]).convert("RGB")
        img = np.array(self.transforms(img))
        lab_img = rgb2lab(img).astype("float32")
        lab_img = transforms.ToTensor()(lab_img)

        # normalize grayscale image to between -1 and 1
        normalized_img = L = (lab_img[[0], ...] / 50.0) - 1.0

        return {'img': normalized_img, 'name': file_name}

    def __len__(self):
        return len(self.input_paths)


class ColorDataset(Dataset):

    def __init__(self, input_path, resize=256, split='train'):
        """
        Input_path: a directory path to color images
        resize: image resolution for image resizing
        split: split data to training or validation
        """
        self.resize = resize
        self.input_path = input_path
        self.split = split

        if split == 'train':
            self.transforms = transforms.Compose(
                    [
                        transforms.Resize((self.resize, self.resize), Image.BICUBIC),
                        transforms.RandomHorizontalFlip()
                    ])
        elif split == 'val':
            self.transforms = transforms.Resize((self.resize, self.resize),  Image.BICUBIC)

    def __getitem__(self, idx):
        """Read RGB images and convert them to Lab images"""
        file_name = self.input_path[idx].split('/')[-1]
        img = Image.open(self.input_path[idx]).convert("RGB")
        img = np.array(self.transforms(img))
        lab_img = rgb2lab(img).astype("float32")
        lab_img = transforms.ToTensor()(lab_img)

        # normalize values to between -1 and 1
        L = (lab_img[[0], ...] / 50.0) - 1.0
        ab = lab_img[[1, 2], ...] / 110.0

        return {'L': L, 'ab': ab, 'name': file_name}

    def __len__(self):
        return len(self.input_path)


