import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 12, 3), nn.BatchNorm2d(12), nn.Dropout(0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(12, 8, 3), nn.BatchNorm2d(8), nn.Dropout(0.1)
        )
        self.transition_block1 = nn.Conv2d(8, 14, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(14, 12, 3), nn.BatchNorm2d(12), nn.Dropout(0.1)
        )
        self.block4 = nn.Conv2d(12, 10, 1)
        self.pool2 = nn.AvgPool2d(10)

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        x = self.pool1(self.transition_block1(x))
        x = F.relu(self.block3(x))
        x = self.pool2(F.relu(self.block4(x)))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.BatchNorm2d(16), nn.Dropout(0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 8, 3), nn.BatchNorm2d(8), nn.Dropout(0.1)
        )
        self.transition_block1 = nn.Conv2d(8, 14, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(14, 12, 3), nn.BatchNorm2d(12), nn.Dropout(0.1)
        )
        self.block4 = nn.Conv2d(12, 10, 1)
        self.pool2 = nn.AvgPool2d(10)

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        x = self.pool1(self.transition_block1(x))
        x = F.relu(self.block3(x))
        x = self.pool2(F.relu(self.block4(x)))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.15)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 12, 3), nn.ReLU(), nn.BatchNorm2d(12), nn.Dropout(0.15)
        )

        self.transition_block1 = nn.Conv2d(12, 24, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(24, 12, 3), nn.ReLU(), nn.BatchNorm2d(12), nn.Dropout(0.15)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(12, 14, 3), nn.ReLU(), nn.BatchNorm2d(14), nn.Dropout(0.15)
        )

        self.block5 = nn.Conv2d(14, 10, 3)
        self.pool2 = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(self.transition_block1(x))
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(F.relu(self.block5(x)))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.15)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 12, 3), nn.ReLU(), nn.BatchNorm2d(12), nn.Dropout(0.15)
        )

        self.transition_block1 = nn.Conv2d(12, 24, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(24, 12, 3), nn.ReLU(), nn.BatchNorm2d(12), nn.Dropout(0.15)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(12, 14, 3), nn.ReLU(), nn.BatchNorm2d(14), nn.Dropout(0.15)
        )

        self.transition_block2 = nn.Conv2d(14, 10, 1)
        self.block5 = nn.Sequential(nn.Conv2d(10, 10, 3), nn.ReLU())
        self.pool2 = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(self.transition_block1(x))
        x = self.block3(x)
        x = self.block4(x)
        x = self.transition_block2(x)
        x = self.pool2(self.block5(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
