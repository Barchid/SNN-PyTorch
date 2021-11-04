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
                               bias=True,
                               dilation=dilation,
                               stride=1)
        self.spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size//2 + dilation - 1,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=True,
                               dilation=dilation,
                               stride=stride)
        self.spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # self.conv2 = ConvSpike(out_channels, out_channels,
        #                        kernel_size, dilation=dilation, stride=stride)

        if stride > 1:
            self.downsample_conv = nn.Conv2d(
                in_channels, out_channels, 1, dilation=dilation, stride=stride)
            self.downsample_spike = snn.Leaky(
                beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)
        else:
            self.downsample_conv = None
            self.downsample_spike = None

    def forward(self, x, mem_1, mem_2, mem_d):
        identity = x

        x = self.conv1(x)
        x, mem_1 = self.spike1(x, mem_1)
        x = self.conv2(x)
        x, mem_2 = self.spike2(x, mem_2)

        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity, mem_d = self.downsample_spike(identity, mem_d)
        else:
            mem_d = None

        if self.connect_f == 'ADD':
            x = x + identity
        elif self.connect_f == 'AND':
            x = x * identity
        else:
            x = x * (1. - identity)

        return x, mem_1, mem_2, mem_d

    def init_leaky(self):
        mem_1 = self.spike1.init_leaky()
        mem_2 = self.spike2.init_leaky()

        if self.downsample_spike is not None:
            mem_d = self.downsample_spike.init_leaky()
        else:
            mem_d = None

        return mem_1, mem_2, mem_d


class ResNet5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timesteps: int):
        super(ResNet5, self).__init__()
        self.timesteps = timesteps

        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7,
                               padding=3,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=True,
                               dilation=1,
                               stride=2)
        self.spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 2
        self.res2_conv1 = nn.Conv2d(64, 64,
                                    kernel_size=3,
                                    padding=1,
                                    # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                    bias=True,
                                    stride=1)
        self.res2_spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res2_conv2 = nn.Conv2d(64, 64,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    stride=2)
        self.res2_spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res2_dconv = nn.Conv2d(64, 64, 1, stride=2)
        self.res2_dspike = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 3
        self.res3_conv1 = nn.Conv2d(64, 128,
                                    kernel_size=3,
                                    padding=1,
                                    # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                    bias=True)
        self.res3_spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res3_conv2 = nn.Conv2d(128, 128,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    stride=2)
        self.res3_spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res3_dconv = nn.Conv2d(64, 128, 1, stride=2)
        self.res3_dspike = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # classifying layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flat = nn.Flatten()

        self.fc = nn.Linear(128, 32, bias=True)
        self.fc_spike = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(
            slope=25), init_hidden=False, output=True)

        self.final = nn.Linear(32, out_channels)

    def forward(self, inputs):
        # resets every LIF neurons
        mem_spike1 = self.spike1.init_leaky()
        mem_res2_spike1 = self.res2_spike1.init_leaky()
        mem_res2_spike2 = self.res2_spike2.init_leaky()
        mem_res2_dspike = self.res2_dspike.init_leaky()
        mem_res3_spike1 = self.res3_spike1.init_leaky()
        mem_res3_spike2 = self.res3_spike2.init_leaky()
        mem_res3_dspike = self.res3_dspike.init_leaky()

        mem_fc_spike = self.fc_spike.init_leaky()

        # spike accumulator to get the prediction
        accumulator = 0.

        for k in range(self.timesteps):
            x = inputs[k, :, :, :, :]
            x = self.conv1(x)
            x, mem_spike1 = self.spike1(x, mem_spike1)

            # residual block 2
            identity = x
            x = self.res2_conv1(x)
            x, mem_res2_spike1 = self.res2_spike1(x, mem_res2_spike1)

            x = self.res2_conv2(x)
            x, mem_res2_spike2 = self.res2_spike2(x, mem_res2_spike2)

            identity = self.res2_dconv(identity)
            identity, mem_res2_dspike = self.res2_dspike(
                identity, mem_res2_dspike)
            x = x + identity

            # residual block 3
            identity = x
            x = self.res3_conv1(x)
            x, mem_res3_spike1 = self.res3_spike1(x, mem_res3_spike1)

            x = self.res3_conv2(x)
            x, mem_res3_spike2 = self.res3_spike2(x, mem_res3_spike2)

            identity = self.res3_dconv(identity)
            identity, mem_res3_dspike = self.res3_dspike(
                identity, mem_res3_dspike)
            x = x + identity

            x = self.avg_pool(x)

            # classifier
            x = self.flat(x)
            x = self.fc(x)
            x, mem_fc_spike = self.fc_spike(x, mem_fc_spike)

            x = self.final(x)

            accumulator += x

        return accumulator


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timesteps: int):
        super(ResNet9, self).__init__()
        self.timesteps = timesteps

        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7,
                               padding=3,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=True,
                               dilation=1,
                               stride=2)
        self.spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 2
        self.res2_conv1 = nn.Conv2d(64, 64,
                                    kernel_size=3,
                                    padding=1,
                                    # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                    bias=True,
                                    stride=1)
        self.res2_spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res2_conv2 = nn.Conv2d(64, 64,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    stride=2)
        self.res2_spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res2_dconv = nn.Conv2d(64, 64, 1, stride=2)
        self.res2_dspike = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 3
        self.res3_conv1 = nn.Conv2d(64, 128,
                                    kernel_size=3,
                                    padding=1,
                                    # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                    bias=True)
        self.res3_spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res3_conv2 = nn.Conv2d(128, 128,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    stride=2)
        self.res3_spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res3_dconv = nn.Conv2d(64, 128, 1, stride=2)
        self.res3_dspike = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 4
        self.res4_conv1 = nn.Conv2d(128, 256,
                                    kernel_size=3,
                                    padding=1,
                                    # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                    bias=True)
        self.res4_spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res4_conv2 = nn.Conv2d(256, 256,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    stride=2)
        self.res4_spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res4_dconv = nn.Conv2d(128, 256, 1, stride=2)
        self.res4_dspike = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 5
        self.res5_conv1 = nn.Conv2d(256, 512,
                                    kernel_size=3,
                                    padding=1,
                                    # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                                    bias=True)
        self.res5_spike1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res5_conv2 = nn.Conv2d(512, 512,
                                    kernel_size=3,
                                    padding=1,
                                    bias=True,
                                    stride=2)
        self.res5_spike2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.res5_dconv = nn.Conv2d(256, 512, 1, stride=2)
        self.res5_dspike = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # classifying layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flat = nn.Flatten()

        self.fc = nn.Linear(512, 64, bias=True)
        self.fc_spike = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(
            slope=25), init_hidden=False, output=True)

        self.final = nn.Linear(64, out_channels, bias=True)

    def forward(self, inputs):
        # resets every LIF neurons
        mem_spike1 = self.spike1.init_leaky()

        mem_res2_spike1 = self.res2_spike1.init_leaky()
        mem_res2_spike2 = self.res2_spike2.init_leaky()
        mem_res2_dspike = self.res2_dspike.init_leaky()

        mem_res3_spike1 = self.res3_spike1.init_leaky()
        mem_res3_spike2 = self.res3_spike2.init_leaky()
        mem_res3_dspike = self.res3_dspike.init_leaky()

        mem_res4_spike1 = self.res4_spike1.init_leaky()
        mem_res4_spike2 = self.res4_spike2.init_leaky()
        mem_res4_dspike = self.res4_dspike.init_leaky()

        mem_res5_spike1 = self.res5_spike1.init_leaky()
        mem_res5_spike2 = self.res5_spike2.init_leaky()
        mem_res5_dspike = self.res5_dspike.init_leaky()

        mem_fc_spike = self.fc_spike.init_leaky()

        # spike accumulator to get the prediction
        accumulator = 0.

        for k in range(self.timesteps):
            x = inputs[k, :, :, :, :]
            x = self.conv1(x)
            x, mem_spike1 = self.spike1(x, mem_spike1)

            # residual block 2
            identity = x
            x = self.res2_conv1(x)
            x, mem_res2_spike1 = self.res2_spike1(x, mem_res2_spike1)

            x = self.res2_conv2(x)
            x, mem_res2_spike2 = self.res2_spike2(x, mem_res2_spike2)

            identity = self.res2_dconv(identity)
            identity, mem_res2_dspike = self.res2_dspike(
                identity, mem_res2_dspike)
            x = x + identity

            # residual block 3
            identity = x
            x = self.res3_conv1(x)
            x, mem_res3_spike1 = self.res3_spike1(x, mem_res3_spike1)

            x = self.res3_conv2(x)
            x, mem_res3_spike2 = self.res3_spike2(x, mem_res3_spike2)

            identity = self.res3_dconv(identity)
            identity, mem_res3_dspike = self.res3_dspike(
                identity, mem_res3_dspike)
            x = x + identity

            # residual block 4
            identity = x
            x = self.res4_conv1(x)
            x, mem_res4_spike1 = self.res4_spike1(x, mem_res4_spike1)

            x = self.res4_conv2(x)
            x, mem_res4_spike2 = self.res4_spike2(x, mem_res4_spike2)

            identity = self.res4_dconv(identity)
            identity, mem_res4_dspike = self.res4_dspike(
                identity, mem_res4_dspike)
            x = x + identity

            # residual block 5
            identity = x
            x = self.res5_conv1(x)
            x, mem_res5_spike1 = self.res5_spike1(x, mem_res5_spike1)

            x = self.res5_conv2(x)
            x, mem_res5_spike2 = self.res5_spike2(x, mem_res5_spike2)

            identity = self.res5_dconv(identity)
            identity, mem_res5_dspike = self.res5_dspike(
                identity, mem_res5_dspike)
            x = x + identity

            x = self.avg_pool(x)

            # classifier
            x = self.flat(x)
            x = self.fc(x)
            x, mem_fc_spike = self.fc_spike(x, mem_fc_spike)

            accumulator += x
        print(torch.count_nonzero(accumulator.clone().detach()))
        output = self.final(accumulator)

        # print(accumulator)
        return torch.sigmoid(output)
