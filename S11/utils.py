import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from test_model import *
import numpy as np
from math import floor, ceil
import torch.nn as nn
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from data_set import *
from PIL import Image


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

visuals = []


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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def wrong_predictions(model):
    fig = plt.figure()
    ax = fig.subplots(2, 5)
    print(len(incorrect_data))
    for i in range(2):
        for j in range(5):
            np_img = incorrect_data[i + j][0]
            np_trans = np.transpose(np_img, (1, 2, 0))
            ax[i][j].imshow(np_trans)
            ax[i][j].set_xlabel(
                f"{classes[incorrect_preds[i+j][0]]}| {classes[original_target[i+j][0]]}"
            )
            fig.show()
            fig.tight_layout()
            plt.tight_layout()
            tens = torch.from_numpy(np_img)
            tens = tens.unsqueeze(dim=0)
            print("before transforms")
            print(np_trans.shape)
            # rgb_img = test_transforms(image=np_trans)["image"]
            # print("after  transfroms")
            # print(rgb_img.shape)
            # rgb_img = np.transpose(rgb_img, (1, 2, 0))
            # print(type(rgb_img))
            # rgb_img = rgb_img / 2 + 0.5
            # rgb_img = rgb_img.detach().cpu().numpy()
            # rgb_img = (rgb_img / 2) + 0.5
            rgb_img = np_trans
            rgb_img = rgb_img / 2 + 0.5
            show_grad_cam_image(model, tens, rgb_img)


def show_kernels(param, kernel_size):
    for i in range(param.shape[0]):
        x_idx, y_idx = floor(param.shape[0] / 2), ceil(param.shape[0] / 2)
        plt.subplot(x_idx, y_idx, i + 1)
        if kernel_size > 3:
            reshaped_tensor = torch.sum(param[i], axis=0)
            plt.imshow(reshaped_tensor.cpu().detach().numpy(), cmap="gray")
        else:
            plt.imshow(np.transpose(param[i].cpu().detach().numpy(), (1, 2, 0)))


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


def show_grad_cam_image(model, input_tesnor, rgb_img):
    print(model.layer3[-1])
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(9)]
    grayscale_cam = cam(input_tensor=input_tesnor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    print(type(visualization))
    print(visualization.shape)
    visuals.append(visualization)


def show_imgs(imgs, labels):
    fig = plt.figure(figsize=(5, 5))
    axs = fig.subplots(4, 7)
    val = 0
    for i in range(4):
        for j in range(7):
            img = (imgs[val] / 2) + 0.5
            npimg = img.numpy()
            trans_npimg = np.transpose(npimg, (1, 2, 0))

            axs[i][j].imshow(trans_npimg)
            axs[i][j].set_xlabel(labels[val].item())
            axs[i][j].tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
                right=False,
            )
            val = val + 1
