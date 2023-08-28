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
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 48, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 56, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(56, 64, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=2, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(48, 56, 3, stride=2, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(56, 32, 3, padding_mode="reflect", padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 36, 3, padding_mode="reflect", padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            nn.Dropout(0.1),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(36, 20, 3, padding_mode="reflect", padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
        )

        self.out = nn.Conv2d(20, 10, 3)
        self.gap = nn.AvgPool2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.out(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
