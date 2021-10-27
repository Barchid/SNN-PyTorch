import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import matplotlib.pyplot as plt
import numpy as np
import itertools


# class UNet(nn.Module):
#     """Some Information about UNet"""

#     def __init__(self, in_channels, height, width):
#         super(UNet, self).__init__()
#         # encoder
#         self.enc1 = ConvSpike(in_channels, 16, 5)  # scale 1
#         self.enc2 = ConvSpike(16, 32, 3, stride=2)  # scale 1/2
#         self.enc3 = ConvSpike(32, 64, 3, stride=2)  # scale 1/4

#         self.enc4 = ConvSpike(64, 128, 3, stride=2)  # scale 1/8

#         # decoder
#         self.up1 = nn.Upsample(scale_factor=2)  # scale 1/4
#         self.dec1 = ConvSpike(128, 64, 3)  # scale 1/4
#         self.up2 = nn.Upsample(scale_factor=2)  # scale 1 /2
#         self.dec2 = ConvSpike(64, 32, 3)  # scale 1/2
#         self.up3 = nn.Upsample(scale_factor=2)  # scale 1
#         self.dec3 = ConvSpike(32, 16, 3)  # scale 1

#         # final fc layers
#         self.flatten = nn.Flatten()
#         self.fc1 = LinearSpike(16 * height * width, 128)

#         self.fc = LinearSpike(128, 4, spike=snn.Leaky(
#             beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), output=False))

#     def forward(self, x):
#         out_enc1 = self.enc1(x)  # scale 1
#         out_enc2 = self.enc2(out_enc1)  # scale 1/2
#         out_enc3 = self.enc3(out_enc2)  # scale 1/4

#         out_latent = self.enc4(out_enc3)  # scale 1/4

#         out_up1 = self.up1(out_latent)  # scale 1/2
#         x = out_up1 + out_enc2

#         out_dec1 = self.dec1(x)

#         out_up2 = self.up2(out_dec1)
#         x = out_up2 + out_enc1

#         out_dec2 = self.dec2(x)

#         return x


class LinearSpike(nn.Sequential):
    def __init__(self, in_channels, out_channels, spike=snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)):
        super(LinearSpike, self).__init__()
        self.add_module('fc', nn.Linear(in_channels, out_channels, bias=False))
        self.add_module('spike', spike)


class ConvSpike(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, batch_norm=False, spike=snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)):
        super(ConvSpike, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d())
        self.add_module('spike', spike)


class SEWBottleneck(nn.Module):
    """Some Information about ResBlockSpike"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        connect_f='ADD'
    ):
        super(SEWBottleneck, self).__init__()

        assert connect_f in ('ADD', 'AND', 'IAND')
        self.connect_f = connect_f

        self.conv1 = ConvSpike(
            in_channels, out_channels, kernel_size, dilation=dilation, stride=1)

        self.conv2 = ConvSpike(
            out_channels, out_channels, kernel_size, dilation=dilation, stride=stride)

        if stride > 1:
            self.downsample = ConvSpike(
                in_channels, out_channels, 1, dilation=dilation, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        if self.connect_f == 'ADD':
            x = x + identity
        elif self.connect_f == 'AND':
            x = x * identity
        else:
            x = x * (1. - identity)

        return x


class ResNet9(nn.Module):
    """Some Information about ResNet9"""

    def __init__(self, in_channels, out_channels):
        super(ResNet9, self).__init__()
        self.layer1 = ConvSpike(in_channels, 64, 7, stride=2)

        self.layer2 = SEWBottleneck(64, 64, 3, stride=2)

        self.layer3 = SEWBottleneck(64, 128, 3, stride=2)

        self.layer4 = SEWBottleneck(128, 256, 3, stride=2)

        self.layer5 = SEWBottleneck(256, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = LinearSpike(512, out_channels, spike=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, output=False))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        out, mem = self.fc(x)

        return out, mem
