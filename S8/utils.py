import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms


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
train_acc1 = []
test_acc = []


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

        pLabels = torch.argmax(output, axis=1)
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
            loss += F.nll_loss(output, target, reduction="sum")
            pred = torch.argmax(output, axis=1)
            acc += (pred == target).sum().item()
        test_acc.append((acc / len(test_loader.dataset)) * 100)
        test_loss = loss / (len(test_loader.dataset))

        print("The Test Accuracy is", test_acc[epoch - 1])
