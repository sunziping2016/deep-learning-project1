# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor


class ImageNetCNN(nn.Module):
    def __init__(self):
        super(ImageNetCNN, self).__init__()
        self.features = nn.Sequential(
            # n x 3 x 64 x 64
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            # n x 64 x 60 x 60
            nn.MaxPool2d(kernel_size=3),
            # n x 64 x 20 x 20
            nn.Conv2d(64, 192, kernel_size=3),
            nn.ReLU(inplace=True),
            # n x 192 x 18 x 18
            nn.MaxPool2d(kernel_size=3),
            # n x 192 x 6 x 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            # n x 1024
            nn.Linear(1024, 100),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def get_transform():
        return ToTensor()
