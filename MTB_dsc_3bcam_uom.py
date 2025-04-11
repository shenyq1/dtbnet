
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from S3_DSConv import DSConv
from torch import cat
cudatpye = "cuda:0"

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Compute query, key, and value
        query = self.conv_query(x)
        key = self.conv_key(x)
        value = self.conv_value(x)

        # Reshape for matrix multiplication
        batch_size, channels, height, width = x.size()
        query = query.view(batch_size, -1, height * width).permute(0, 2, 1)
        key = key.view(batch_size, -1, height * width)

        # Compute attention weights
        attn = F.softmax(torch.bmm(query, key), dim=-1)

        # Apply attention to the value
        out = torch.bmm(value.view(batch_size, -1, height * width), attn.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Scale and apply gamma
        out = self.gamma * out + x

        return out

class Cross_SelfAttention(nn.Module):
    def __init__(self, in_channels):#in_channels实际是输入特征图的通道数的一半
        super(Cross_SelfAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.ptconv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        #获得x的第二个维度的长度
        n = x.size()[1]//2# n = inchannel
        x1 = x[:,0:n,:,:]
        x2 = x[:,n:,:,:]
        # Compute query, key, and value
        query1 = self.conv_query(x1)#b*n*h*w->b*n/8*h*w
        key1 = self.conv_key(x1)
        value1 = self.conv_value(x1)#b*n*h*w->b*n*h*w
        query2 = self.conv_query(x2)
        key2 = self.conv_key(x2)#
        value2 = self.conv_value(x2)

        # Reshape for matrix multiplication
        batch_size, channels, height, width = x1.size()
        query1 = query1.view(batch_size, -1, height * width).permute(0, 2, 1)#  (b,n/8,h,w)->(b,h*w,n/8)
        key1 = key1.view(batch_size, -1, height * width)#(b,n/8,h,w)->(b,n/8,h*w)
        query2 = query2.view(batch_size, -1, height * width).permute(0, 2, 1)
        key2 = key2.view(batch_size, -1, height * width)


        # Compute attention weights
        attn1 = F.softmax(torch.bmm(query1, key1), dim=-1)#(b,h*w,n/8)*(b,n/8,h*w)->(b,h*w,h*w) 一个batch中的所有像素点的注意力权重
        attn2 = F.softmax(torch.bmm(query2, key2), dim=-1)

        # Apply attention to the value
        out11 = torch.bmm(value1.view(batch_size, -1, height * width), attn1.permute(0, 2, 1))#(b,n,h*w)*(b,h*w,h*w)->(b,n,h*w)
        out11 = out11.view(batch_size, channels, height, width)
        out12 = torch.bmm(value2.view(batch_size, -1, height * width), attn1.permute(0, 2, 1))
        out12 = out12.view(batch_size, channels, height, width)
        out21 = torch.bmm(value1.view(batch_size, -1, height * width), attn2.permute(0, 2, 1))
        out21 = out21.view(batch_size, channels, height, width)
        out22 = torch.bmm(value2.view(batch_size, -1, height * width), attn2.permute(0, 2, 1))
        out22 = out22.view(batch_size, channels, height, width)
        
        #所有的out拼接起来，维度经过点卷积变成原来的维度
        out11 = torch.cat([out11, out12], dim=1)#(b,2n,h,w)
        out11 = self.ptconv(out11)
        out21 = torch.cat([out21, out22], dim=1)
        out21 = self.ptconv(out21)
        out = torch.cat([out11, out21], dim=1)#(b,2n,h,w)

        # Scale and apply gamma
        out = self.gamma * out + x#(b,2n,h,w)

        return out


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
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False),
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
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)      
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

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

class MTB_dsc_3bcam_uom(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(MTB_dsc_3bcam_uom, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels//2, base_c,kernel_size=7)#输入是六波段两个时相
        self.conv0x = DSConv(
            in_channels//2,
            base_c,
            3,
            1,
            0,
            True,
            cudatpye,
        )
        self.conv0y = DSConv(
            in_channels//2,
            base_c,
            3,
            1,
            1,
            True,
            cudatpye,
        )
        self.conv0 = EncoderConv(3 * base_c, base_c)#三个卷积结果进行拼接后的卷积层 3n----n
        self.ptconv0 = CatConv(base_c * 2, base_c)#ptconv0是将两个时相的特征图进行拼接，然后卷积 2n----n
        self.cross_attention0 = Cross_SelfAttention(base_c)#输入是单个时相的通道数

        self.down1 = nn.MaxPool2d(2, stride=2)
        self.doubleconv1 = DoubleConv(base_c, base_c * 2,kernel_size=5)
        #down包括池化和两次卷积
        self.conv1x = DSConv(
            base_c,
            base_c * 2,
            3,
            1,
            0,
            True,
            cudatpye,
        )
        self.conv1y = DSConv(
            base_c,
            base_c * 2,
            3,
            1,
            1,
            True,
            cudatpye,
        )
        self.conv1 = EncoderConv(6 * base_c, base_c * 2)
        self.ptconv1 = CatConv(base_c * 4, base_c * 2)
        self.cross_attention1 = Cross_SelfAttention(base_c * 2)#输入是单个时相的通道数

        self.down2 = nn.MaxPool2d(2, stride=2)
        self.doubleconv2 = DoubleConv(base_c*2, base_c * 4)
        self.conv2x = DSConv(
            base_c*2,
            base_c * 4,
            3,
            1,
            0,
            True,
            cudatpye,
        )
        self.conv2y = DSConv(
            base_c*2,
            base_c * 4,
            3,
            1,
            1,
            True,
            cudatpye,
        )
        self.conv2 = EncoderConv(12 * base_c, base_c * 4)
        self.ptconv2 = CatConv(base_c * 8, base_c * 4)
        self.cross_attention2 = Cross_SelfAttention(base_c * 4)#输入是单个时相的通道数

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
            cudatpye,
        )
        self.conv3y = DSConv(
            base_c*4,
            base_c * 8,
            3,
            1,
            1,
            True,
            cudatpye,
        )
        self.conv3 = EncoderConv(24 * base_c, base_c * 8)
        self.ptconv3 = CatConv(base_c * 16, base_c * 8)
        self.cross_attention3 = Cross_SelfAttention(base_c * 8)#输入是单个时相的通道数

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
            cudatpye,
        )
        self.conv4y = DSConv(
            base_c*8,
            base_c * 16 // factor,
            3,
            1,
            1,
            True,
            cudatpye,
        )
        self.conv4 = EncoderConv(48 * base_c // factor, base_c * 16 // factor)
        self.ptconv4 = CatConv(base_c * 32 // factor, base_c * 16 // factor)
        self.cross_attention4 = Cross_SelfAttention(base_c * 16 // factor)#输入是单个时相的通道数

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)#up包括上采样和两次卷积
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.unet = UNet(in_channels=4, num_classes=2, base_c=32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#layer1
        midlayer = []   
        x_ori = x
        x11 = self.in_conv(x[:,0:3,:,:]) #in --- base   X-time-layer

        x21 = self.in_conv(x[:,3:6,:,:])

        x1 = self.ptconv0(torch.cat([x11, x21], dim=1))#`2base --- base`
        midlayer.append(x1)
#layer2
        x11 = self.down1(x11)
        x12 = self.doubleconv1(x11)
        # x12x = self.conv1x(x11)
        # x12y = self.conv1y(x11)
        # x12 = self.conv1(torch.cat([x12, x12x, x12y], dim=1))
        x21 = self.down1(x21)
        x22 = self.doubleconv1(x21)
        # x22x = self.conv1x(x21)
        # x22y = self.conv1y(x21)
        # x22 = self.conv1(torch.cat([x22, x22x, x22y], dim=1))
        x2 = self.ptconv1(torch.cat([x12, x22], dim=1))#`4base --- 2base`
        midlayer.append(x2)
#layer3
        x12 = self.down2(x12)
        x13 = self.doubleconv2(x12)

        x22 = self.down2(x22)
        x23 = self.doubleconv2(x22)

        x3 = self.cross_attention2(torch.cat([x13, x23], dim=1))
        x3 = torch.cat([x13, x23], dim=1)
        x3 = self.ptconv2(x3)#8base --- base128
        midlayer.append(x3)
#layer4
        x13 = self.down3(x13)
        x14 = self.doubleconv3(x13)

        x23 = self.down3(x23)
        x24 = self.doubleconv3(x23)

        x4 = torch.cat([x14, x24], dim=1)
        # x4 = self.cross_attention3(torch.cat([x14, x24], dim=1))#16base --- base256
        x4 = self.ptconv3(x4)#base256
        midlayer.append(x4)
#layer5
        x14 = self.down4(x14)#256
        x15 = self.doubleconv4(x14)#256
        x24 = self.down4(x24)#256
        x25 = self.doubleconv4(x24)#256
        # x5 = self.cross_attention4(torch.cat([x15, x25], dim=1))#512 
        x5 = torch.cat([x15, x25], dim=1)
        x5 = self.ptconv4(x5)#256
        midlayer.append(x5)

        x = self.up1(x5, x4)#
        midlayer.append(x)
        x = self.up2(x, x3)
        midlayer.append(x)
        x = self.up3(x, x2)
        midlayer.append(x)
        x = self.up4(x, x1)
        midlayer.append(x)
        logits = self.out_conv(x)
        midlayer.append(logits)
        logits2 = torch.cat([logits, x_ori[:,[0],:,:],x_ori[:,[3],:,:]], dim=1)
        logits2 = self.unet(logits2)#用于改善边缘效果

        return {"out": logits, "aux": logits2, "mid":midlayer}

