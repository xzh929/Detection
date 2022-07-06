from pathlib import Path
import os
import sys
import yaml
import argparse
from tqdm import tqdm
from torch import nn
import torch
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.dataloaders import create_dataloader
from utils.general import colorstr, check_yaml, one_cycle
from utils.loss import ComputeLoss
from yolov3 import YoloV3

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp, opt):
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    hyp['label_smoothing'] = opt.label_smoothing
    imgsz = opt.imgsz
    gs = 32
    batch_size = opt.batch_size
    single_cls = opt.single_cls
    workers = opt.workers
    epochs = opt.epochs
    train_path = r"D:\code\py\datasets\mydata\images\train"
    module_path = r"module/yolov3.pt"
    summary = SummaryWriter("logs")

    model = YoloV3().cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
    if opt.optimizer == "Adam":
        optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g[1]})
    del g

    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    compute_loss = ComputeLoss(hyp)
    for epoch in range(epochs):
        model.train()
        sum_loss = 0.
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    file=sys.stdout)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.cuda(non_blocking=True).float() / 255
            with torch.cuda.amp.autocast():
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.cuda())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            sum_loss += loss.item()

        scheduler.step()
        avg_loss = sum_loss / len(train_loader)
        print("epoch:{} loss:{}".format(epoch + 1, avg_loss))
        summary.add_scalar("loss", avg_loss, epoch)
        checkpoint = {"model": model,
                      "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict()}
        if epoch % 20 == 0:
            torch.save(checkpoint, module_path)
            print("save success")

    torch.cuda.empty_cache()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=416, help='train, val image size (pixels)')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    opt.hyp = check_yaml(opt.hyp)
    train(opt.hyp, opt)
