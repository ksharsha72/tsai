import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class Cifar(Dataset):
    def __init__(self, dataset, transform=None) -> None:
        self.dataset = dataset
        self.transforms = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return (image, label)


train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=10, p=0.2),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_height=16,
            min_width=16,
            min_holes=1,
            fill_value=[0.49139968, 0.48215827, 0.44653124],
        ),
        ToTensorV2(),
        A.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768],
        ),
    ]
)

test_transforms = A.Compose(
    [
        ToTensorV2(),
        A.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768],
        ),
    ]
)


def get_summary(model, set_device=False, input_size=(3, 32, 32)):
    device = None
    if set_device:
        is_cuda = torch.cuda.is_available()
        print(is_cuda)
        device = "cuda" if is_cuda else "cpu"
        device = torch.device(device)
        model = model.to(device)
    summary(model, input_size)
    return model, device


train_loss = []
test_loss = []
train_acc = []
test_acc = []
incorrect_preds = []
incorrect_data = []
original_target = []
layer_weights = []
layers = []
org_data = []
classes = (
    "Airplane",
    "Automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def train(model, device, train_loader, optimizer, epoch, **kwargs):
    model.train()
    acc = 0
    acc1 = 0
    epoch_loss = 0
    processed = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        if device != (None or "cpu"):
            data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pLabels = torch.argmax(output, dim=1)
        acc += (pLabels == target).sum().item()
        processed += len(target)
        epoch_loss += loss.item()
        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*acc/processed:0.2f}"
        )

    acc = acc / processed * 100
    train_acc.append(acc)
    train_loss.append(epoch_loss / len(train_loader.dataset))


def test(model, device, test_loader, epoch):
    model.eval()
    acc = 0
    loss = 0
    processed = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            if device != (None or "cpu"):
                data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()
            pred = torch.argmax(output, dim=1)
            acc += (pred == target).sum().item()
        test_acc.append((acc / len(test_loader.dataset)) * 100)
        test_loss.append(loss / (len(test_loader.dataset)))

        if test_acc[-1] >= sorted(test_acc)[-1]:
            global incorrect_preds, incorrect_data, original_target
            torch.save(model.state_dict(), "./best_model.pth")
            incorrect_preds.append(
                pred[torch.where(~(pred == target))[0].cpu().numpy()]
            )

            original_target.append(
                target[torch.where(~(pred == target))[0].cpu().numpy()]
            )
            incorrect_data.append(data[torch.where(~(pred == target))[0]].cpu().numpy())
        print("The Test Accuracy is", test_acc[epoch - 1])


def show_kernels(param, kernel_size):
    for i in range(param.shape[0]):
        x_idx, y_idx = floor(param.shape[0] / 2), ceil(param.shape[0] / 2)
        plt.subplot(x_idx, y_idx, i + 1)
        if kernel_size > 3:
            reshaped_tensor = torch.sum(param[i], axis=0)
            plt.imshow(reshaped_tensor.cpu().detach().numpy(), cmap="gray")
        else:
            plt.imshow(np.transpose(param[i].cpu().detach().numpy(), (1, 2, 0)))


# def helper(param):
#  if param[1] <= 3:
#     for i in range(param.shape[0]):
#         x_idx, y_idx = floor(param.shape[0] / 2), ceil(
#             param.shape[0] / 2
#         )
#         plt.subplot(x_idx, y_idx, i + 1)
#         plt.imshow(
#             np.transpose(
#                 param[i].cpu().detach().numpy(), (1, 2, 0)
#             )
#         )
#     else:
#         for i in range(param.shape[0]):
#             reshaped_tensor = torch.sum(param[0], axis=0)
#             x_idx, y_idx = floor(param.shape[0] / 2), ceil(
#                 param.shape[0] / 2
#             )

#             plt.subplot(x_idx, y_idx, i + 1)
#             plt.imshow(
#                 np.transpose(
#                     param[i].cpu().detach().numpy(),
#                     (1, 2, 0),
#                 )
#             )


def plot_kernels(model):
    for child in model.children():
        val = 0
        if type(child) == nn.Sequential:
            for child1 in child:
                if type(child1) == nn.Conv2d:
                    for idx, param in enumerate(child1.parameters()):
                        if idx == 0:
                            if param.shape[1] <= 3:
                                show_kernels(param, kernel_size=3)
                            else:
                                show_kernels(param, kernel_size=4)


def wrong_predictions():
    fig = plt.figure()
    ax = fig.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            np_img = incorrect_data[i + j][0]
            print(np_img.shape)
            np_trans = np.transpose(np_img, (1, 2, 0))
            ax[i][j].imshow(np_trans)
            ax[i][j].set_xlabel(
                f"{classes[incorrect_preds[i+j][0]]}| {classes[original_target[i+j][0]]}"
            )
            fig.show()
            fig.tight_layout()
            plt.tight_layout()
