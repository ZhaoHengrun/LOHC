import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .basic_module import SpectralNorm, Self_Attn, GatedConv2dWithActivation, GatedDeConv2dWithActivation, \
    SNConvWithActivation, get_pad


def padding(input, output):
    input_h, input_w = input.shape[2], input.shape[3]
    output_h, output_w = output.shape[2], output.shape[3]
    if input_h != output_h:
        pad = nn.ReplicationPad2d(padding=(0, 0, 0, input_h - output_h))
        output = pad(output)
    if input_w != output_w:
        pad = nn.ReplicationPad2d(padding=(0, input_w - output_w, 0, 0))
        output = pad(output)
    return output


class DeepFill_v2_Generator(torch.nn.Module):
    """
    Inpaint generator, input should be 4*256*256, where 3*256*256 is the masked image,
    1*256*256 for mask
    """

    def __init__(self, in_channel=4):
        super(DeepFill_v2_Generator, self).__init__()
        channel_num = 32
        self.coarse_net = nn.Sequential(
            # input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(in_channel, channel_num, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(channel_num, 2 * channel_num, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * channel_num, 2 * channel_num, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample to 64
            GatedConv2dWithActivation(2 * channel_num, 4 * channel_num, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=16,
                                      padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            # Self_Attn(4*channel_num, 'relu'),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4 * channel_num, 2 * channel_num, 3, 1, padding=get_pad(128, 3, 1)),
            # Self_Attn(2*channel_num, 'relu'),
            GatedConv2dWithActivation(2 * channel_num, 2 * channel_num, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * channel_num, channel_num, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(channel_num, channel_num // 2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(channel_num//2, 'relu'),
            GatedConv2dWithActivation(channel_num // 2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(in_channel, channel_num, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(channel_num, channel_num, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(channel_num, 2 * channel_num, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample
            GatedConv2dWithActivation(2 * channel_num, 2 * channel_num, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            # Self_Attn(4*channel_num, 'relu'),
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, dilation=16,
                                      padding=get_pad(64, 3, 1, 16))
        )
        self.refine_attn = Self_Attn(4 * channel_num, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * channel_num, 4 * channel_num, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4 * channel_num, 2 * channel_num, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2 * channel_num, 2 * channel_num, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * channel_num, channel_num, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(channel_num, channel_num // 2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(channel_num, 'relu'),
            GatedConv2dWithActivation(channel_num // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

    def forward(self, imgs, masks):
        # Coarse
        masked_imgs = imgs * (1 - masks) + masks
        input_imgs = torch.cat([masked_imgs, masks], dim=1)
        x = self.coarse_net(input_imgs)
        x = padding(input_imgs, x)
        x = torch.clamp(x, -1., 1.)
        coarse_x = x
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        input_imgs = torch.cat([masked_imgs, masks], dim=1)
        x = self.refine_net(input_imgs)
        x = self.refine_attn(x)
        x = self.refine_upsample_net(x)
        x = padding(input_imgs, x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x


class DeepFill_v2_Discriminator(nn.Module):
    def __init__(self, in_ch=4):
        super(DeepFill_v2_Discriminator, self).__init__()
        channel_num = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(in_ch, 2 * channel_num, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2 * channel_num, 4 * channel_num, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(8, 5, 2)),
            Self_Attn(8 * channel_num, 'relu'),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8 * channel_num * 2 * 2, 1)


    def forward(self, x):
        x = self.discriminator_net(x)
        x = x.view((x.size(0), -1))
        return x


class DeepFill_v2_Discriminator_Human(nn.Module):
    def __init__(self, in_ch=4):
        super(DeepFill_v2_Discriminator_Human, self).__init__()
        channel_num = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(in_ch, 2 * channel_num, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2 * channel_num, 4 * channel_num, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(8, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(4, 5, 2)),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(2, 5, 2)),
            Self_Attn(8 * channel_num, 'relu'),
            SNConvWithActivation(8 * channel_num, 8 * channel_num, 4, 2, padding=get_pad(1, 5, 2)),
        )
        self.linear = nn.Linear(8 * channel_num * 2 * 2, 1)


    def forward(self, x):
        x = self.discriminator_net(x)
        x = x.view((x.size(0), -1))
        return x