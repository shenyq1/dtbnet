from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from S3_DSConv import DSConv
from torch import cat


class EncoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class CatConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(CatConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class MTUNet_dsc(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(MTUNet_dsc, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels//2, base_c)#输入是六波段两个时相
        self.conv0x = DSConv(
            in_channels//2,
            base_c,
            3,
            1,
            0,
            True,
            "cuda:1",
        )
        self.conv0y = DSConv(
            in_channels//2,
            base_c,
            3,
            1,
            1,
            True,
            "cuda:1",
        )
        self.conv0 = EncoderConv(3 * base_c, base_c)#三个卷积结果进行拼接后的卷积层 3n----n
        self.ptconv0 = CatConv(base_c * 2, base_c)#ptconv0是将两个时相的特征图进行拼接，然后卷积 2n----n

        self.down1 = nn.MaxPool2d(2, stride=2)
        self.doubleconv1 = DoubleConv(base_c, base_c * 2)
        #down包括池化和两次卷积
        self.conv1x = DSConv(
            base_c,
            base_c * 2,
            3,
            1,
            0,
            True,
            "cuda:1",
        )
        self.conv1y = DSConv(
            base_c,
            base_c * 2,
            3,
            1,
            1,
            True,
            "cuda:1",
        )
        self.conv1 = EncoderConv(6 * base_c, base_c * 2)
        self.ptconv1 = CatConv(base_c * 4, base_c * 2)

        self.down2 = nn.MaxPool2d(2, stride=2)
        self.doubleconv2 = DoubleConv(base_c*2, base_c * 4)
        self.conv2x = DSConv(
            base_c*2,
            base_c * 4,
            3,
            1,
            0,
            True,
            "cuda:1",
        )
        self.conv2y = DSConv(
            base_c*2,
            base_c * 4,
            3,
            1,
            1,
            True,
            "cuda:1",
        )
        self.conv2 = EncoderConv(12 * base_c, base_c * 4)
        self.ptconv2 = CatConv(base_c * 8, base_c * 4)

        # self.down3 = Down(base_c * 4, base_c * 8)
        self.down3 = nn.MaxPool2d(2, stride=2)
        self.doubleconv3 = DoubleConv(base_c*4, base_c * 8)
        self.conv3x = DSConv(
            base_c*4,
            base_c * 8,
            3,
            1,
            0,
            True,
            "cuda:1",
        )
        self.conv3y = DSConv(
            base_c*4,
            base_c * 8,
            3,
            1,
            1,
            True,
            "cuda:1",
        )
        self.conv3 = EncoderConv(24 * base_c, base_c * 8)
        self.ptconv3 = CatConv(base_c * 16, base_c * 8)

        factor = 2 if bilinear else 1
        # self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.down4 = nn.MaxPool2d(2, stride=2)
        self.doubleconv4 = DoubleConv(base_c*8, base_c * 16 // factor)
        self.conv4x = DSConv(
            base_c*8,
            base_c * 16 // factor,
            3,
            1,
            0,
            True,
            "cuda:1",
        )
        self.conv4y = DSConv(
            base_c*8,
            base_c * 16 // factor,
            3,
            1,
            1,
            True,
            "cuda:1",
        )
        self.conv4 = EncoderConv(48 * base_c // factor, base_c * 16 // factor)
        self.ptconv4 = CatConv(base_c * 32 // factor, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)#up包括上采样和两次卷积
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#layer1
        x11 = self.in_conv(x[:,0:3,:,:]) #in --- base   X-time-layer
        x11x = self.conv0x(x[:,0:3,:,:]) #in --- base
        x11y = self.conv0y(x[:,0:3,:,:])#in --- base
        x11 = self.conv0(torch.cat([x11, x11x, x11y], dim=1))#3base --- base
        x21 = self.in_conv(x[:,3:6,:,:])
        x21x = self.conv0x(x[:,3:6,:,:])
        x21y = self.conv0y(x[:,3:6,:,:])
        x21 = self.conv0(torch.cat([x21, x21x, x21y], dim=1))
        x1 = self.ptconv0(torch.cat([x11, x21], dim=1))#`2base --- base`
#layer2
        x11 = self.down1(x11)
        x12 = self.doubleconv1(x11)
        x12x = self.conv1x(x11)
        x12y = self.conv1y(x11)
        x12 = self.conv1(torch.cat([x12, x12x, x12y], dim=1))
        x21 = self.down1(x21)
        x22 = self.doubleconv1(x21)
        x22x = self.conv1x(x21)
        x22y = self.conv1y(x21)
        x22 = self.conv1(torch.cat([x22, x22x, x22y], dim=1))
        x2 = self.ptconv1(torch.cat([x12, x22], dim=1))#`4base --- 2base`
#layer3
        x12 = self.down2(x12)
        x13 = self.doubleconv2(x12)
        x13x = self.conv2x(x12)
        x13y = self.conv2y(x12)
        x13 = self.conv2(torch.cat([x13, x13x, x13y], dim=1))
        x22 = self.down2(x22)
        x23 = self.doubleconv2(x22)
        x23x = self.conv2x(x22)
        x23y = self.conv2y(x22)
        x23 = self.conv2(torch.cat([x23, x23x, x23y], dim=1))
        x3 = self.ptconv2(torch.cat([x13, x23], dim=1))# `8base --- 4base`
#layer4
        x13 = self.down3(x13)
        x14 = self.doubleconv3(x13)
        x14x = self.conv3x(x13)
        x14y = self.conv3y(x13)
        x14 = self.conv3(torch.cat([x14, x14x, x14y], dim=1))
        x23 = self.down3(x23)
        x24 = self.doubleconv3(x23)
        x24x = self.conv3x(x23)
        x24y = self.conv3y(x23)
        x24 = self.conv3(torch.cat([x24, x24x, x24y], dim=1))
        x4 = self.ptconv3(torch.cat([x14, x24], dim=1))# `16base --- 8base`
#layer5
        x14 = self.down4(x14)
        x15 = self.doubleconv4(x14)
        x15x = self.conv4x(x14)
        x15y = self.conv4y(x14)
        x15 = self.conv4(torch.cat([x15, x15x, x15y], dim=1))
        x24 = self.down4(x24)
        x25 = self.doubleconv4(x24)
        x25x = self.conv4x(x24)
        x25y = self.conv4y(x24)#
        x25 = self.conv4(torch.cat([x25, x25x, x25y], dim=1))# 
        x5 = self.ptconv4(torch.cat([x15, x25], dim=1))# `32base --- 16base`
        


        
        # x12 = self.down1(x11)
        # x12 = self.doubleconv1(x12)
        # x22 = self.down1(x21)
        # x2 = torch.cat([x12, x22], dim=1)
        # x2 = self.ptconv1(x2)
        # x13 = self.down2(x12)
        # x23 = self.down2(x22)
        # x3 = torch.cat([x13, x23], dim=1)
        # x3 = self.ptconv2(x3)
        # x14 = self.down3(x13)
        # x24 = self.down3(x23)
        # x4 = torch.cat([x14, x24], dim=1)
        # x4 = self.ptconv3(x4)
        # x15 = self.down4(x14)
        # x25 = self.down4(x24)
        # x5 = torch.cat([x15, x25], dim=1)
        # x5 = self.ptconv4(x5)
        # x1 = self.in_conv(x[:,0:3,:,:])
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}

