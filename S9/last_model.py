import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BaseModel, self).__init__(*args, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        # self.dil_conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, dilation=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Dropout(0.1),
        # )

        self.conv3 = nn.Sequential(nn.Conv2d(64, 72, 3, stride=2, padding=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 72, 3, groups=72, padding=1),
            nn.Conv2d(72, 56, 1),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(56, 48, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1),
        )

        self.dil_conv5 = nn.Sequential(
            nn.Conv2d(56, 48, 3, dilation=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(48, 56, 3, stride=2, padding=2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(56, 56, 3, groups=56, padding=1),
            nn.Conv2d(56, 28, 1),
            nn.ReLU(),
            nn.BatchNorm2d(28),
            nn.Dropout(0.1),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(28, 28, 3, groups=28, padding=2),
            nn.Conv2d(28, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 52, 3, stride=2, padding=2),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(52, 52, 3, groups=52, padding=1),
            nn.Conv2d(52, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 42, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(42),
            nn.Dropout(0.1),
        )

        self.dil_conv11 = nn.Sequential(
            nn.Conv2d(32, 42, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(42),
            nn.Dropout(0.1),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(42, 36, 3), nn.ReLU(), nn.BatchNorm2d(36), nn.Dropout(0.1)
        )

        self.out = nn.Conv2d(36, 10, 1)
        self.gap = nn.AvgPool2d(5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # + self.dil_conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) + self.dil_conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x) + self.dil_conv11(x)
        x = self.conv12(x)
        x = self.out(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
