import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *
from networks import Flatten, get_pad, GatedConv, SNConvWithActivation, Self_Attn


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, use_bias=True, dilation_rate=1):
        super(ConvRelu, self).__init__()
        self.conv = GatedConv(in_channels, out_channels, kernel, stride=1, padding=padding, bias=use_bias, dilation=dilation_rate)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=use_bias, dilation=dilation_rate),
        #     nn.ReLU()
        # )

    def forward(self, x):
        output = self.conv(x)
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, use_bias=True, dilation_rate=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)

    def forward(self, x):
        output = self.conv(x)
        return output


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DenseNet(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(DenseNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # (3, 2, 1, 1, 1, 1)
        self.conv_relu_1 = ConvRelu(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel=3, padding=3,
                                    dilation_rate=3)
        self.conv_relu_2 = ConvRelu(in_channels=self.in_channels * 2, out_channels=self.out_channels,
                                    kernel=3, padding=2,
                                    dilation_rate=2)
        self.conv_relu_3 = ConvRelu(in_channels=self.in_channels * 3, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=1)
        self.conv_relu_4 = ConvRelu(in_channels=self.in_channels * 4, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=1)
        self.conv_relu_5 = ConvRelu(in_channels=self.in_channels * 5, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=1)
        self.conv_relu_6 = ConvRelu(in_channels=self.in_channels * 6, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=1)
        self.channel_squeeze = Conv(in_channels=self.in_channels * 7, out_channels=self.out_channels,
                                    kernel=1, padding=0)

    def forward(self, x):
        t = self.conv_relu_1(x)  # 64
        _t = torch.cat([x, t], dim=1)  # 128

        t = self.conv_relu_2(_t)
        _t = torch.cat([_t, t], dim=1)  #

        t = self.conv_relu_3(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_4(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_5(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_6(_t)
        _t = torch.cat([_t, t], dim=1)
        _t = self.channel_squeeze(_t)
        return _t


class CFN(nn.Module):

    def __init__(self, in_ch=9, out_ch=3):
        super(CFN, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # self.conv_in = ConvRelu(in_channels=in_ch, out_channels=64, kernel=3, padding=1, dilation_rate=1)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )
        self.densenet = DenseNet(64, 64)
        # self.conv_out = ConvRelu(in_channels=64, out_channels=out_ch, kernel=3, padding=1, dilation_rate=1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.densenet(x)
        x = self.conv_out(x)
        return x


class DeepFill_v2_Discriminator(nn.Module):
    def __init__(self):
        super(DeepFill_v2_Discriminator, self).__init__()
        channel_num = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(4, 2 * channel_num, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2 * channel_num, 4 * channel_num, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(8, 5, 2)),
            Self_Attn(8 * channel_num, 'relu'),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8 * channel_num * 2 * 2, 1)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        x = self.discriminator_net(x)  # [8, 256, 2, 2]
        x = x.view((x.size(0), -1))  # [8, 1024]
        return x

class DeepFill_v2_DiscriminatorS(nn.Module):
    def __init__(self):
        super(DeepFill_v2_DiscriminatorS, self).__init__()
        channel_num = 8
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(4, 2 * channel_num, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2 * channel_num, 4 * channel_num, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(4, 5, 2)),
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        x = self.discriminator_net(x)  # [8, 256, 2, 2]
        x = x.view((x.size(0), -1))  # [8, 1024]
        return x