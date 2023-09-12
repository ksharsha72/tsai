import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict as od


class CustomResnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CustomResnet, self).__init__(*args, **kwargs)

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.pool4 = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.conv1(x)
        x_res1 = self.res1(x)
        x = x + x_res1
        x = self.conv2(x)
        x = self.conv3(x)
        x_res2 = self.res2(x)
        x = x + x_res2
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
