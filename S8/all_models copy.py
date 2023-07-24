import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(
        self,
        norm="bn",
        in_ch=3,
        out_ch=4,
        kernel_size=3,
        padding=1,
        group_size=2,
        usepool=True,
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

        if usepool:
            self.pool = nn.MaxPool2d(2, 2)


class CIFARnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CIFARnet, self).__init__(*args, **kwargs)
