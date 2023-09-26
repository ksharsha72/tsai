from train import *
from data_set import *
from utils import *
from test import *

from torchvision import datasets

train_data = CustomDataSet(
    datasets.CIFAR10("../../data", train=True, download=True),
    transform=train_transfroms,
)

test_data = CustomDataSet(
    datasets.CIFAR10("../../data", train=False, download=True),
    transform=test_transforms,
)


batch_size = 512

kwargs = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 2,
    "pin_memory": True,
}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
