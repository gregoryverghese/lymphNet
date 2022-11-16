""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,mode='bilinear',align_corners=False)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,stride=1,dilation=1,padding=1):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,dilation=dilation,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_mr(nn.Module):
      
      def __init__(self,in_channels,out_channels):
          super(Down_mr,self).__init__()
          out_channels=int(out_channels/3)
          self.ds0=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels))
          self.ds1=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels,dilation=2,padding=2,stride=2))
          self.ds2=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels,dilation=4,padding=4,stride=4))
      def forward(self,x):
          out1=self.ds0(x)
          out2=F.interpolate(self.ds1(x),(out1.shape[2],out1.shape[3]),mode='bilinear',align_corners=False)
          out3=F.interpolate(self.ds2(x),(out1.shape[2],out1.shape[3]),mode='bilinear',align_corners=False)
          #print(x.shape,out1.shape,out2.shape,out3.shape)
          return torch.cat((out1,out2,out3),1)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,dilation=1):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False,dilation=1):
        super(Up,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = F.interpolate#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1,scale_factor=2,mode='bilinear',align_corners=False)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.interpolate(x1,(x2.shape[2],x2.shape[3]),mode='bilinear',align_corners=False)
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
