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


class Baseline5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timesteps: int):
        super(Baseline5, self).__init__()
        self.timesteps = timesteps

        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7,
                               padding=3,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=True,
                               dilation=1,
                               stride=2)
        self.spike1 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 2
        self.conv2 = nn.Conv2d(64, 64,
                               kernel_size=3,
                               padding=1,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=True,
                               stride=2)
        self.spike2 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.conv3 = nn.Conv2d(64, 128,
                               kernel_size=3,
                               padding=1,
                               bias=True,
                               stride=2)
        self.spike3 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # residual block 3
        self.conv4 = nn.Conv2d(128, 256,
                               kernel_size=3,
                               padding=1,
                               # no bias because it is not bio-plausible (and hard to impl in neuromorphic hardware)
                               bias=True,
                               stride=2)
        self.spike4 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        self.conv5 = nn.Conv2d(256, 512,
                               kernel_size=3,
                               padding=1,
                               bias=True,
                               stride=2)
        self.spike5 = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=False)

        # classifying layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flat = nn.Flatten()
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512, out_channels, bias=True)
        self.fc_spike = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(
            slope=25), init_hidden=False, output=True)

        # self.final = nn.Linear(128, out_channels, bias=True)

    def forward(self, inputs):
        # resets every LIF neurons
        mem_spike1 = self.spike1.init_leaky()
        mem_spike2 = self.spike2.init_leaky()
        mem_spike3 = self.spike3.init_leaky()
        mem_spike4 = self.spike4.init_leaky()
        mem_spike5 = self.spike5.init_leaky()

        mem_fc_spike = self.fc_spike.init_leaky()

        # mem accumulator to get the prediction
        accumulator = []

        for k in range(self.timesteps):
            x = inputs[k, :, :, :, :]
            x = self.conv1(x)
            x, mem_spike1 = self.spike1(x, mem_spike1)

            x = self.conv2(x)
            x, mem_spike2 = self.spike2(x, mem_spike2)

            x = self.conv3(x)
            x, mem_spike3 = self.spike3(x, mem_spike3)

            x = self.conv4(x)
            x, mem_spike4 = self.spike4(x, mem_spike4)

            x = self.conv5(x)
            x, mem_spike5 = self.spike5(x, mem_spike5)

            x = self.avg_pool(x)

            # classifier
            x = self.flat(x)
            x = self.dropout(x)
            x = self.fc(x)
            x, mem_fc_spike = self.fc_spike(x, mem_fc_spike)

            # x = self.final(x)

            accumulator.append(mem_fc_spike)

        return accumulator
