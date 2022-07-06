from torchvision.models import resnet50
from torchvision import transforms as t
from torch.optim import SGD, lr_scheduler
import torch
x = torch.randn(2,3,32,32)
#调用预权重模型
net = resnet50(pretrained=False)
print(net)
# y = net(x)
# print(y.shape)
# #图片尺寸resize224
# trans = t.Compose([t.ToTensor(), t.Resize(224)])
#
# opt = SGD(net.parameters(), lr=0.001, momentum=0.9)
# #学习率优化策略：余弦退火
# schedular = lr_scheduler.CosineAnnealingLR(opt,T_max=5)
# #接在每一个train_loader循环完
# schedular.step()

