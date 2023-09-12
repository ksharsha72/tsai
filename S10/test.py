import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from tqdm import tqdm


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    acc = 0
    loss = 0
    processed = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            if device != (None or "cpu"):
                data, target = data.to(device), target.to(device)
            output = model(data)
            # loss += F.nll_loss(output, target, reduction="sum").item()
            loss += criterion(output, target, reduction="sum").item()
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
