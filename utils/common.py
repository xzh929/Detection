from torch import nn
from yolov3 import YoloV3
import torch
from torch.optim import SGD


class CreateModule(nn.Module):
    def __init__(self, weight):
        super().__init__()
        ckpt = torch.load(weight)
        self.model = YoloV3().cuda()
        self.model.load_state_dict(ckpt["model_state_dict"])

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)
        self.optimizer = SGD(g[2], lr=0.01, momentum=0.937, nesterov=True)
        self.optimizer.add_param_group({'params': g[0], 'weight_decay': 0.0005})
        self.optimizer.add_param_group({'params': g[1]})
        del g
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def forward(self, x):
        pred = self.model(x)
        return pred


if __name__ == '__main__':
    x = torch.randn(2, 3, 416, 416)
    model = CreateModule(r"module/yolov3.pt")
    pred = model(x)
    print(pred[0].shape, pred[1].shape, pred[2].shape)
