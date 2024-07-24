# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CoordAtt_3d(nn.Module):
    def __init__(self, inp, oup, reduction=32, threshold=8):     # dafault=32, dafault=8
        super(CoordAtt_3d, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1,None))

        mip = max(threshold, inp // reduction)
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c,d, h, w = x.size()
        x_d = self.pool_d(x)
        x_h = self.pool_h(x).permute(0,1,3,2,4)
        x_w = self.pool_w(x).permute(0,1,4,2,3)

        y = torch.cat([x_h, x_w,x_d], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_d,x_h, x_w = torch.split(y, [d,h, w], dim=2)
        x_h =x_h.permute(0,1,3,2,4)
        x_w =x_w.permute(0,1,3,4,2)

        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h * a_d
        return out#todo 不用注意力机制试一下


# x_3d = torch.randn(2, 64,16, 224, 224)  # b, c, h, w
# ca_model_3d = CoordAtt_3d(inp=64,oup=64)
# y=ca_model_3d(x_3d)

# x_2d = torch.randn(2, 64, 224, 224)
# att_2d = CoordAtt(inp=64,oup=64)
# y=att_2d(x_2d)
# print(y.shape)
