from torch import nn
import torch
from torch.nn import functional as f
from thop import profile, clever_format

nn.Dropout2d
class Convolutional(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, stride, padding):
        super(Convolutional, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, ksize, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Residual(nn.Module):
    def __init__(self, out_channel):
        super(Residual, self).__init__()
        self.layer = nn.Sequential(
            Convolutional(out_channel, out_channel // 2, 1, 1, 0),
            Convolutional(out_channel // 2, out_channel, 3, 1, 1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out + x


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            Convolutional(in_channel, out_channel, 3, 2, 1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Convolutional_set(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convolutional_set, self).__init__()
        self.layer = nn.Sequential(
            Convolutional(in_channel, out_channel, 1, 1, 0),
            Convolutional(out_channel, in_channel, 3, 1, 1),
            Convolutional(in_channel, out_channel, 1, 1, 0),
            Convolutional(out_channel, in_channel, 3, 1, 1),
            Convolutional(in_channel, out_channel, 1, 1, 0),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x):
        return f.interpolate(x, scale_factor=2)


class YoloV3(nn.Module):
    def __init__(self):
        super(YoloV3, self).__init__()
        self.Darknet53_layer_52 = nn.Sequential(
            Convolutional(3, 32, 3, 1, 1),  # 416*416
            DownSample(32, 64),  # 208*208

            Residual(64),
            DownSample(64, 128),  # 104*104

            Residual(128),
            Residual(128),
            DownSample(128, 256),  # 52*52

            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256)
        )
        self.Darknet53_layer_26 = nn.Sequential(
            DownSample(256, 512),  # 26*26
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512)
        )
        self.Darknet53_layer_13 = nn.Sequential(
            DownSample(512, 1024),  # 13*13
            Residual(1024),
            Residual(1024),
            Residual(1024),
            Residual(1024)
        )
        self.conv_set1 = Convolutional_set(1024, 512)
        self.predict_one = nn.Sequential(
            Convolutional(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 21, 1, 1)  # 13*13
        )
        self.up_26 = nn.Sequential(
            Convolutional(512, 512, 1, 1, 0),
            UpSample()  # 26*26
        )
        self.conv_set2 = Convolutional_set(1024, 512)
        self.predict_two = nn.Sequential(
            Convolutional(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 21, 1, 1)  # 26*26
        )
        self.up_52 = nn.Sequential(
            Convolutional(512, 256, 1, 1, 0),
            UpSample()  # 52*52
        )
        self.predict_three = nn.Sequential(
            Convolutional_set(512, 256),
            Convolutional(256, 512, 3, 1, 1),
            nn.Conv2d(512, 21, 1, 1)  # 52*52
        )

    def forward(self, x):
        out_52 = self.Darknet53_layer_52(x)
        out_26 = self.Darknet53_layer_26(out_52)
        out_13 = self.Darknet53_layer_13(out_26)
        out_13_set = self.conv_set1(out_13)
        pre_one = self.predict_one(out_13_set)
        pre_one = pre_one.reshape(-1, 3, 7, 13, 13).permute(0, 1, 4, 3, 2)
        up_26 = self.up_26(out_13_set)
        out_26_set = self.conv_set2(torch.cat((out_26, up_26), dim=1))
        pre_two = self.predict_two(out_26_set).reshape(-1, 3, 7, 26, 26).permute(0, 1, 4, 3, 2)
        up_52 = self.up_52(out_26_set)
        pre_three = self.predict_three(torch.cat((up_52, out_52), dim=1)).reshape(-1, 3, 7, 52, 52).permute(0, 1, 4, 3,
                                                                                                           2)
        return pre_three, pre_two, pre_one


if __name__ == '__main__':
    x = torch.randn(2, 3, 416, 416)
    net = YoloV3()
    # flops, params = profile(net, inputs=(x,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)
    # print(net)
    y1, y2, y3 = net(x)
    print(y1.shape, y2.shape, y3.shape)

