import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


train_transforms = transforms.Compose(
    [
        transforms.RandomRotation((-7, 7)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            incorrect_data.append(data[incorrect_preds[-1]].cpu().numpy())
        print("The Test Accuracy is", test_acc[epoch - 1])


def plot_kernels(model):
    for child in model.children():
        val = 0
        if type(child) == nn.Sequential:
            for child1 in child:
                if type(child1) == nn.Conv2d:
                    print("_______________this is true_______________")
                    print(child1.parameters == child1.weight)
                    if val > 5:
                        for idx, param in enumerate(child1.parameters()):
                            if (idx == 0) and (param.shape[1] <= 3):
                                for i in range(param.shape[0]):
                                    plt.imshow(param[i].cpu().detach().numpy())
                                    plt.show()
                                    break
                            elif (idx == 0) and param.shape[1] > 3:
                                for i in range(param.shape[0]):
                                    npimg = (
                                        torch.sum(param[0], 0).cpu().detach().numpy()
                                    )
                                    plt.imshow(npimg)
                                    plt.show()
                                    break
                    val = val + 1
                break
        break


def wrong_predictions():
    fig = plt.figure()
    ax = fig.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            if len(incorrect_data) > 10:
                np_img = incorrect_data[i + j]
                print(np_img.shape)
                # np_trans = np.transpose(np_img, (1, 2, 0))
                ax[i][j].plot(np_img)
                ax[i][j].set_xlabel(f"{incorrect_preds[i+j]} \ f{original_target[i+j]}")
