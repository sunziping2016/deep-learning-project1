# -*- coding: utf-8 -*-
import os

import torch.utils.data as data
from PIL import Image


class ImageNetDataset(data.Dataset):
    def __init__(self, root: str, is_test: bool = False,
                 transform=None, target_transform=None):
        with open(root) as f:
            lines = f.readlines()
        root = os.path.dirname(root)
        if is_test:
            self.images = []
            for line in lines:
                line = line.strip()
                if line:
                    self.images.append(os.path.join(root, line))
        else:
            self.images, self.labels = [], []
            for line in lines:
                line = line.strip()
                if line:
                    image, label = line.split(' ')
                    self.images.append(os.path.join(root, image))
                    self.labels.append(int(label))
        self.is_test = is_test
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        with open(self.images[index], 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        image = image
        if self.transform is not None:
            image = self.transform(image)
        if self.is_test:
            return image
        else:
            label = self.labels[index]
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label

    def __len__(self):
        return len(self.images)
