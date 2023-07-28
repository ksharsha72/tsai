import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 24, 3),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2),
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
        self.layer4_dil = nn.Sequential(
            nn.Conv2d(32, 56, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
            nn.Conv2d(56, 64, 1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 72, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.Dropout(0.1),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(72, 56, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
        )
        self.dw_sep1 = nn.Sequential(
            nn.Conv2d(56, 56, 3, groups=56, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(56),
            nn.Dropout(0.1),
            nn.Conv2d(56, 48, 1),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer12 = nn.Sequential(nn.Conv2d(32, 10, 3, stride=2))
        self.out = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) + self.layer4_dil(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.dw_sep1(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
