from tqdm import tqdm
from utils import *

batch_loss = []


def train(
    train_loader, model, optimizer, critireon, device, epoch, scheduler, **kwargs
):
    model.train()
    pbar = tqdm(train_loader)
    acc = 0
    epoch_loss = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        if device != (None or "cpu"):
            data = data.to(device)
            target = target.to(device)
        pred = model(data, target)
        loss = critireon(pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        lrs.append(get_lr(optimizer))
        pLabels = torch.argmax(pred, dim=1)

        acc += (pLabels == target).sum().item()
        processed += len(target)
        epoch_loss += loss.item()
        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*acc/processed:0.2f}"
        )

    acc = acc / processed * 100
    train_acc.append(acc)
    train_loss.append(epoch_loss / len(train_loader.dataset))
