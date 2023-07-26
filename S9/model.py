import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        self.dil1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.dw_sep1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, groups=64, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.Conv2d(128, 32, 1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 32, 3, dilation=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 48, 3),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(48, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        self.dw_sep2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, groups=64, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.Conv2d(128, 48, 1),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(32, 10, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1),
        )

        self.out = nn.AvgPool2d(5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        dilout = self.dil1(x)
        print(dilout.shape)
        print(self.dw_sep1(x).shape)
        x = dilout + self.dw_sep1(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.dw_sep2(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.out(x)
        return F.log_softmax(x)