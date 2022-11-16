""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .multi_block import *


class UNet_multi(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_multi, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels,63)  
        self.down1 = Down_mr(63,120)
        self.down2 = Down_mr(120,270)
        self.down3 = Down_mr(270,360)
        self.down4 = Down_mr(360,480)
        self.down5 = Down_mr(480,480)
        self.up1 = Up(480+480,480, bilinear)
        self.up2 = Up(480+360,360, bilinear)
        self.up3 = Up(360+270,270, bilinear)
        self.up4 = Up(270+120,120, bilinear)
        self.up5 = Up(120+63,63, bilinear)
        self.outc = OutConv(63, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print(x1.shape,'INshape')
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        #print("====================================UP done==========")
        #print(x.shape,x.shape)
        #assert 3==5,"End"
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
