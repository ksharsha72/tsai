import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from math import floor, ceil


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


# from torch_lr_finder import LRFinder

# model = CustomResnet().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
# critireon = nn.CrossEntropyLoss()
# lr_finder = LRFinder(model, optimizer, critireon, device="cuda")
# lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
# lr_finder.plot()
# lr_finder.reset()


# from torch.optim.lr_scheduler import OneCycleLR

# EPOCHS = 24

# scheduler = OneCycleLR(
#     optimizer,
#     max_lr="that is given by that of the lr_finder",
#     steps_per_epoch=len(train_loader),
#     epochs=EPOCHS,
#     pct_start=5 / EPOCHS,
#     div_factor=100,
#     three_phase=False,
#     final_div_factor=100,
#     anneal_strategy="linear",
# )


# for epoch in range(EPOCHS):
#     print("EPOCH", epoch)
#     train(model,device,train_loader,optimizer,epoch,scheduler,critireon)
#     test(model,test_device,test_laoder,critireon)
