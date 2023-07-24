from typing import Any, Optional
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(
        self, norm="bn", in_ch=3, out_ch=4, kernel_size=3, padding=1, group_size=2
    ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=padding
        )
        if norm == "bn":
            self.n1 = nn.BatchNorm2d(out_ch)
        elif norm == "ln":
            self.n1 = nn.GroupNorm(1, out_ch)
        elif norm == "gn":
            self.n1 = nn.GroupNorm(2, out_ch)

        self.conv2 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=padding
        )
        if norm == "bn":
            self.n2 = nn.BatchNorm2d(out_ch)
        elif norm == "ln":
            self.n2 = nn.GroupNorm(1, out_ch)
        elif norm == "gn":
            self.n2 = nn.GroupNorm(2, out_ch)

        self.conv3 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=padding
        )
        if norm == "bn":
            self.n3 = nn.BatchNorm2d(out_ch)
        elif norm == "ln":
            self.n3 = nn.GroupNorm(1, out_ch)
        elif norm == "gn":
            self.n3 = nn.GroupNorm(2, out_ch)

    def __call__(self, x, num_layers):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.n1(x)
        if num_layers >= 2:
            x = self.conv2(x)
            x = F.relu(x)
            x = self.n2(x)
        if num_layers >= 3:
            x = self.conv3(x)
            x = F.relu(x)
            x = self.n2(x)
        return
