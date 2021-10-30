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

        # self.conv1 = ConvSpike(
        #     in_channels, out_channels, kernel_size, dilation=dilation, stride=1)

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size//2 + dilation - 1,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=False,
                               dilation=dilation,
                               stride=1)
        self.spike1 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size//2 + dilation - 1,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=False,
                               dilation=dilation,
                               stride=stride)
        self.spike2 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)

        # self.conv2 = ConvSpike(out_channels, out_channels,
        #                        kernel_size, dilation=dilation, stride=stride)

        if stride > 1:
            self.downsample_conv = nn.Conv2d(
                in_channels, out_channels, 1, dilation=dilation, stride=stride)
            self.downsample_spike = snn.Leaky(
                beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)
        else:
            self.downsample_conv = None
            self.downsample_spike = None

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.spike1(x)
        x = self.conv2(x)
        x = self.spike2(x)

        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_spike(identity)

        if self.connect_f == 'ADD':
            x = x + identity
        elif self.connect_f == 'AND':
            x = x * identity
        else:
            x = x * (1. - identity)

        return x

    def init_leaky(self):
        self.spike1.init_leaky()
        self.spike2.init_leaky()

        if self.downsample_spike is not None:
            self.downsample_spike.init_leaky()


class ResNet9(nn.Module):
    """Some Information about ResNet9"""

    def __init__(self, in_channels, out_channels):
        super(ResNet9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7,
                               padding=3,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=False,
                               dilation=1,
                               stride=2)
        self.spike1 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)

        self.layer2 = SEWBottleneck(64, 64, 3, stride=2)

        self.layer3 = SEWBottleneck(64, 128, 3, stride=2)

        self.layer4 = SEWBottleneck(128, 256, 3, stride=2)

        self.layer5 = SEWBottleneck(256, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flat = nn.Flatten()

        self.fc = nn.Linear(512, out_channels, bias=False)
        self.fc_spike = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(
            slope=25), init_hidden=True, output=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.spike1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avg_pool(x)

        # classifier
        x = self.flat(x)
        x = self.fc(x)
        out, mem = self.fc_spike(x)

        return out, mem


class ResNet5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timesteps: int):
        super(ResNet5, self).__init__()
        self.timesteps = timesteps

        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7,
                               padding=3,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=False,
                               dilation=1,
                               stride=2)
        self.spike1 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)

        self.layer2 = SEWBottleneck(64, 64, 3, stride=2)

        self.layer3 = SEWBottleneck(64, 128, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flat = nn.Flatten()

        self.fc = nn.Linear(128, 32, bias=False)
        self.fc_spike = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(
            slope=25), init_hidden=True, output=True)

        self.non_spike_fc = nn.Linear(32, out_channels)

    def forward(self, inputs):
        # resets every LIF neurons
        self.spike1.init_leaky()
        self.layer2.init_leaky()
        self.layer3.init_leaky()
        self.fc_spike.init_leaky()

        # spike accumulator to get the prediction
        accumulator = 0.

        for k in self.timesteps:
            x = inputs[k, :, :, :]
            x = self.conv1(x)
            x = self.spike1(x)

            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avg_pool(x)

            # classifier
            x = self.flat(x)
            x = self.fc(x)
            x, _ = self.fc_spike(x)

            x = self.non_spike_fc(x)

            accumulator += x

        return accumulator


class SmallCNN(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(SmallCNN, self).__init__()
        beta = 0.5
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.add_module('conv1', nn.Conv2d(in_channels, 12, 5))
        self.add_module('max1', nn.MaxPool2d(2))
        self.add_module('spike1', snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True))
        self.add_module('conv2', nn.Conv2d(12, 64, 5))
        self.add_module('max2', nn.MaxPool2d(2))
        self.add_module('spike2', snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True))
        self.add_module('flattent', nn.Flatten())
        self.add_module('linear', nn.Linear(64*4*4, 10))
        self.add_module('spike_linear', snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True, output=True))


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
