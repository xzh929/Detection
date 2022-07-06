from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms as t
from torch import nn
from utils.general import one_cycle
from tqdm import tqdm
import warnings

transforms = t.Compose([t.ToTensor()])
dataset = CIFAR10(root=r"D:\data", train=True, transform=transforms)
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
model = resnet18().cuda()
warnings.filterwarnings("ignore")
loss_fun = nn.CrossEntropyLoss()
g = [], [], []  # optimizer parameter groups
bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
for v in model.modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
        g[2].append(v.bias)
    if isinstance(v, bn):  # weight (no decay)
        g[1].append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
        g[0].append(v.weight)
optimizer = Adam(model.parameters())
lf = one_cycle(1, 0.01, 300)
scheduler = lr_scheduler.LambdaLR(optimizer, lf)
scaler = GradScaler()
for epoch in range(300):
    model.train()
    sum_loss = 0.
    tbar = enumerate(tqdm(train_loader))
    optimizer.zero_grad()
    for i, (img, label) in tbar:
        with autocast():
            out = model(img.cuda())
            loss = loss_fun(out, label.cuda())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        sum_loss += loss.item()

    scheduler.step()
    avg_loss = sum_loss / len(train_loader)
    print("epoch:{} loss:{}".format(epoch + 1, avg_loss))
