from torch.utils.data.dataset import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

mean = [0.49139968, 0.48215827, 0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]

train_transforms = A.Compose(
    [
        A.Normalize(mean=mean, std=std, always_apply=True),
        A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_REFLECT, p=0.1, always_apply=True),
        A.RandomCrop(32, 32, p=0.2, always_apply=True),
        A.HorizontalFlip(p=0.1),
        A.CoarseDropout(1, 8, 8, 1, 8, 8, fill_value=mean),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


class CustomLoader(Dataset):
    def __init__(self, dataset, transforms=None) -> None:
        if transforms != None:
            self.transform = transforms
        # self.data = dataset.data
        # self.labels = dataset.labels
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # image = self.data[index]
        # label = self.labels[index]
        image, label = self.dataset[index]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
