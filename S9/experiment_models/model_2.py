import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 24, 3),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1),
        )
        self.layer2_dil = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=2, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 72, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.Dropout(0.1),
        )
        self.dw_sep1 = nn.Sequential(
            nn.Conv2d(72, 72, 3, groups=72, padding=2),
            nn.Conv2d(72, 48, 1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(48, 56, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(56, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(32, 36, 3),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            nn.Dropout(0.1),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(36, 24, 3),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1),
        )
        self.layer12 = nn.Sequential(nn.Conv2d(24, 10, 3))
        self.out = nn.AvgPool2d(3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) + self.layer2_dil(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.dw_sep1(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
