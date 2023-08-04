import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Net, self).__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 56, 3, stride=2, padding=5, padding_mode="reflect"),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(56, 168, 3, groups=56, padding_mode="reflect", padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(168, 56, 1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(56, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 24, 3, stride=2, padding=4, padding_mode="reflect"),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(24, 36, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer7_2 = nn.Sequential(
            nn.Conv2d(24, 36, 3, dilation=2, padding=2, padding_mode="reflect"),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(36, 56, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(56, 32, 3, stride=2, padding=4, padding_mode="reflect"),
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(24, 10, 3),
        )

        self.gap = nn.AvgPool2d(7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x) + self.layer7_2(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
