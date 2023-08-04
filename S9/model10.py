import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 72, 3, stride=2, padding=1, padding_mode="reflect")
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 48, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 56, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(48, 56, 3, dilation=2, padding=2, padding_mode="reflect"),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(56, 48, 3, stride=2, padding=1, padding_mode="reflect")
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 28, 3, padding=1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(28, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.conv10 = nn.Sequential(nn.Conv2d(24, 10, 3))

        self.gap = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) + self.conv5_1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
