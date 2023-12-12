import torch.nn as nn
import torch
from torchvision.models import resnet152, inception_v3
from torch.nn.functional import interpolate
import torch.nn.functional as F
from networks import Flatten, get_pad, GatedConv, SNConvWithActivation, Self_Attn
from utils import save_image_tensor


def padding(img, size):
    b, c, h, w = img.shape  # h = 33
    if h % size != 0 or w % size != 0:
        border_h = size - (h % size)  # border_h = 16-(33%16) = 15
        border_w = size - (w % size)
        pad = nn.ZeroPad2d(padding=(0, border_w, 0, border_h))  # left right up bottom
        img = pad(img)
        padding_flag = True
    else:
        padding_flag = False
    return img, h, w, padding_flag


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
        # self.linear = nn.Linear(8 * channel_num * 2 * 2, 1)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        x = self.discriminator_net(x)  # [8, 256, 2, 2]
        x = x.view((x.size(0), -1))  # [8, 1024]
        return x


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.conv1 = GatedConv(in_ch, mid_ch, 3, 1, padding=1)
        self.conv2 = GatedConv(mid_ch, out_ch, 3, 1, padding=1)
        self.conv_skip = GatedConv(in_ch, out_ch, 3, 1, padding=1)

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + x_skip
        return x


class conv_0_j(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_0_j, self).__init__()
        self.conv = GatedConv(in_ch, out_ch, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, factor, ch):
        super(upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.pixel_shuffle = nn.PixelShuffle(factor)
        # self.conv_1 = GatedConv(ch, ch * factor * factor, 3, 1, padding=1)
        # self.conv_2 = GatedConv(ch, ch, 3, 1, padding=1)

    def forward(self, x):
        x = self.up(x)
        # x = self.conv_1(x)
        # x = self.pixel_shuffle(x)
        # x = self.conv_2(x)
        return x


class GatedUNetPP(nn.Module):

    def __init__(self, in_ch=64, out_ch=3):
        super(GatedUNetPP, self).__init__()

        n1 = 64
        filters = [n1, n1, n1, n1, n1]  # [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up_1_0 = upsample(2, filters[1])
        self.Up_2_0 = upsample(2, filters[2])
        self.Up_3_0 = upsample(2, filters[3])
        self.Up_4_0 = upsample(2, filters[4])

        self.Up_1_1 = upsample(2, filters[1])
        self.Up_2_1 = upsample(2, filters[2])
        self.Up_1_2 = upsample(2, filters[1])
        self.Up_3_1 = upsample(2, filters[3])
        self.Up_2_2 = upsample(2, filters[2])
        self.Up_1_3 = upsample(2, filters[1])

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv0_1_extra = conv_0_j(filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv0_2_extra = conv_0_j(filters[0], filters[0])

        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv0_3_extra = conv_0_j(filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv0_4_extra = conv_0_j(filters[0], filters[0])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_ch, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        x, h, w, padding_flag = padding(x)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up_1_0(x1_0)], 1))
        x0_1 = self.conv0_1_extra(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up_2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up_1_1(x1_1)], 1))
        x0_2 = self.conv0_2_extra(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up_3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up_2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up_1_2(x1_2)], 1))
        x0_3 = self.conv0_3_extra(x0_3)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up_4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up_3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up_2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up_1_3(x1_3)], 1))
        x0_4 = self.conv0_4_extra(x0_4)

        output = self.final(x0_4)
        if padding_flag is True:
            output = output[:, :, :h, :w]
        return output


class ResBlock(nn.Module):
    def __init__(self, ch=64, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        return identity + out * self.res_scale


class GatedResBlock(nn.Module):
    def __init__(self, ch=32, res_scale=1):
        super(GatedResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = GatedConv(ch, ch, 3, 1, padding=1)
        self.conv2 = GatedConv(ch, ch, 3, 1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        return identity + out * self.res_scale


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            GatedConv(in_channels, mid_channels, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            GatedConv(mid_channels, out_channels, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GatedUNet(nn.Module):
    def __init__(self, n_channels=64, n_classes=3, bilinear=True):
        super(GatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # 256
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 64
        x4 = self.down3(x3)  # 32
        x5 = self.down4(x4)  # 16
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class GeneratorL(nn.Module):
    def __init__(self):
        super(GeneratorL, self).__init__()
        self.conv_in = GatedConv(4, 64, 3, 1, padding=1)
        self.conv_in_mae = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.mae_conv_256 = nn.Sequential(ResBlock())
        self.mae_conv_128 = nn.Sequential(ResBlock())
        self.mae_conv_64 = nn.Sequential(ResBlock(), ResBlock())
        self.mae_conv_32 = nn.Sequential(ResBlock())
        self.mae_conv_16 = nn.Sequential(ResBlock())
        self.fc_noise = nn.Sequential(
            nn.Linear(1 * 64 * 64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4096),
            nn.LeakyReLU(),
        )

        self.add_noise = nn.Sequential(
            GatedConv(128, 64, 3, 1, padding=1),
            nn.LeakyReLU()
        )

        self.merge_1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.merge_2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.merge_3 = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.merge_4 = nn.Sequential(
            nn.Conv2d(512 + 64, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.merge_5 = nn.Sequential(
            nn.Conv2d(512 + 64, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.n_channels = 64
        self.bilinear = True
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, 3)

    def forward(self, x, mae_pred):
        # mask = x[:, -1, :, :].unsqueeze(1)
        b, _, _, _ = x.shape
        x, h, w, padding_flag = padding(x, 32)
        x = self.conv_in(x)
        x_mae_64 = self.conv_in_mae(mae_pred)
        x_mae_64 = self.mae_conv_64(x_mae_64)  # 64
        x_mae_32 = F.interpolate(x_mae_64, scale_factor=0.5, mode='bilinear')
        x_mae_32 = self.mae_conv_32(x_mae_32)
        x_mae_16 = F.interpolate(x_mae_32, scale_factor=0.5, mode='bilinear')
        x_mae_16 = self.mae_conv_16(x_mae_16)
        x_mae_128 = F.interpolate(x_mae_64, scale_factor=2, mode='bilinear')
        x_mae_128 = self.mae_conv_128(x_mae_128)
        x_mae_256 = F.interpolate(x_mae_128, scale_factor=2, mode='bilinear')
        x_mae_256 = self.mae_conv_256(x_mae_256)

        noise = torch.randn([x.shape[0], 1 * 64 * 64]).cuda()
        noise = self.fc_noise(noise)
        noise = noise.view([b, 64, 64]).unsqueeze(1)  # [b, 1, 64, 64]
        noise = F.interpolate(noise, size=(256, 256), mode='bilinear')  # [b, 1, 256, 256]
        noise = noise.repeat(1, 64, 1, 1)  # [b, 32, 256, 256]
        x = torch.cat([x, noise], dim=1)
        x = self.add_noise(x)

        x1 = self.inc(x)  # 256
        x1 = torch.cat([x1, x_mae_256], dim=1)
        x1 = self.merge_1(x1)

        x2 = self.down1(x1)  # 128
        x2 = torch.cat([x2, x_mae_128], dim=1)
        x2 = self.merge_2(x2)

        x3 = self.down2(x2)  # 64
        x3 = torch.cat([x3, x_mae_64], dim=1)
        x3 = self.merge_3(x3)

        x4 = self.down3(x3)  # 32
        x4 = torch.cat([x4, x_mae_32], dim=1)
        x4 = self.merge_4(x4)

        x5 = self.down4(x4)  # 16
        x5 = torch.cat([x5, x_mae_16], dim=1)
        x5 = self.merge_5(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if padding_flag is True:
            x = x[:, :, :h, :w]
        return x
