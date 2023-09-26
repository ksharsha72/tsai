import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models import *
from torch_lr_finder import LRFinder

epoch_batch_loss = {}
epoch_batch_acc = {}
train_loss = []
train_acc = []
lrs = []


def train(
    model, device, train_loader, criterion, optimizer, scheduler, epoch, **kwargs
):
    acc = 0
    epoch_loss = 0
    processed = 0
    epoch_batch_loss[epoch] = {}
    epoch_batch_acc[epoch] = {}

    model.train()
    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        batch_acc = 0
        batch_loss = 0
        epoch_batch_acc[epoch][batch_idx] = []
        epoch_batch_loss[epoch][batch_idx] = []

        if device != (None or "cpu"):
            data = (data.to(device),)
            target = target.to(device)
        pred = model(data)
        loss = criterion(target, pred)
        loss.backward()
        optimizer.step()
        scheduler.step()
        lrs.append(get_lr["lr"])

        pLabels = torch.argmax(pred, dim=1)
        batch_acc = (target == pLabels).sum().item()
        epoch_batch_acc[epoch][batch_idx] = batch_acc
        epoch_batch_loss[epoch][batch_idx] = loss.item()
        epoch_loss += loss.item()
        processed += len(target)
        acc += batch_acc
        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*acc/processed:0.2f}"
        )

    acc = (acc / processed) * 100
    final_loss = epoch_loss / len(train_loader.dataset)
    train_acc.append(acc)
    train_loss.append(final_loss)


from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = 24


def get_lr_finder(optimizer, train_loader, critireon, device):
    model = ResNet18().to(device)
    lr_finder = LRFinder(model, optimizer, critireon, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    _, min_lr = lr_finder.plot(suggest_lr=True)
    lr_finder.reset()
    return min_lr


def get_scheduler(optimizer, train_loader, critireon, device):
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=get_lr_finder(optimizer, critireon, train_loader, device),
        steps_per_epoch=len(train_loader),
        pct_start=3 / EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy="linear",
    )
    return scheduler


def get_optimizer(model, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    return optimizer


critireon = nn.CrossEntropyLoss(reduction="sum")
