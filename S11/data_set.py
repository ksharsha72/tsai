import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset
import numpy as np

mean = (0.49139968, 0.48215827, 0.44653124)
std = (0.49139968, 0.48215827, 0.44653124)

train_transfroms = A.Compose(
    [
        A.Normalize(mean=mean, std=std),
        A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_REFLECT, always_apply=True),
        A.RandomCrop(32, 32, always_apply=True),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_height=16,
            min_width=16,
            fill_value=mean,
            p=0.1,
        ),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


class CustomDataSet(Dataset):
    def __init__(self, dataset, transform=None) -> None:
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
