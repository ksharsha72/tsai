from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(0.1),
        )

        self.transition_block1 = nn.Conv2d(32, 56, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(56, 32, 3),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(0.1),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(0.1),
        )

        self.transition_block2 = nn.Conv2d(16, 24, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block6 = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(0.1),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(0.1),
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(16, 12, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, 12),
            nn.Dropout(0.1),
        )

        self.pool3 = nn.AvgPool2d(6)
        self.block9 = nn.Conv2d(12, 10, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(self.transition_block1(x))
        x = self.block3(x)
        x = self.block4(x)
        x = x + self.block5(x)
        x = self.pool2(self.transition_block2(x))
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool3(x)
        x = self.block9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
