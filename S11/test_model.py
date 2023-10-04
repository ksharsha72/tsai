import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


test_acc = []
test_loss = []
incorrect_preds = []
incorrect_data = []
original_target = []
data_without_transforms = []


def test_model(model, device, test_loader, criterion, epoch):
    model.eval()
    pbar = tqdm(test_loader)
    processed = 0
    epoch_loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            batch_acc = 0
            if device != None or "cpu":
                data = data.to(device)
                target = target.to(device)

            pred = model(data)
            loss = criterion(pred, target)
            pLabels = torch.argmax(pred, dim=1)
            batch_acc = (target == pLabels).sum().item()
            processed += len(target)

            epoch_loss += loss.item()
            acc += batch_acc

        acc = (acc / processed) * 100
        test_acc.append(acc)
        test_loss.append(epoch_loss / len(test_loader.dataset))

        if test_acc[-1] >= sorted(test_acc)[-1]:
            global incorrect_preds, incorrect_data, original_target
            torch.save(model.state_dict(), "./best_model.pth")

            incorrect_preds.append(
                pLabels[torch.where(~(pLabels == target))[0].cpu().numpy()]
            )

            original_target.append(
                target[torch.where(~(pLabels == target))[0].cpu().numpy()]
            )
            incorrect_data.append(
                data[torch.where(~(pLabels == target))[0]].cpu().numpy()
            )
        print("The Test Accuracy is", test_acc[epoch - 1])
