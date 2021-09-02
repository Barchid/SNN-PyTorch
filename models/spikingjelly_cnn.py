import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
import numpy as np
from spikingjelly.clock_driven import functional, surrogate, layer, neuron


class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class SpikingJellyNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(SpikingJellyNet.conv3x3(2, channels))
        conv.append(nn.MaxPool2d(2, 2))
        for i in range(4):
            conv.extend(SpikingJellyNet.conv3x3(channels, channels))
            conv.append(nn.MaxPool2d(2, 2))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False),
            neuron.LIFNode(
                tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(0.5),
            nn.Linear(channels * 2 * 2, 110, bias=False),
            neuron.LIFNode(
                tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.vote(self.fc(self.conv(x[0])))
        for t in range(1, x.shape[0]):
            out_spikes += self.vote(self.fc(self.conv(x[t])))
        return out_spikes / x.shape[0]

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.LIFNode(
                tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        ]


try:
    import cupy

    class CextNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            conv = []
            conv.extend(CextNet.conv3x3(2, channels))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            for i in range(4):
                conv.extend(CextNet.conv3x3(channels, channels))
                conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(2),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(
                    nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False)),
                neuron.MultiStepLIFNode(
                    tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(
                    nn.Linear(channels * 2 * 2, 110, bias=False)),
                neuron.MultiStepLIFNode(
                    tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            )
            self.vote = VotingLayer(10)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
            out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
            return self.vote(out_spikes.mean(0))

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
                neuron.MultiStepLIFNode(
                    tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            ]

    class CextNet2(nn.Module):
        def __init__(self, channels: int, T: int, b: int):
            super().__init__()
            self.T, self.b = T, b
            self.conv2d = nn.Sequential(
                nn.Flatten(0, 1),
                *CextNet2.block_2d(self, 2, channels),
                nn.MaxPool2d(2, 2),
                *CextNet2.block_2d(self, channels, channels),
                nn.MaxPool2d(2, 2),
                *CextNet2.block_2d(self, channels, channels),
                nn.MaxPool2d(2, 2),
                *CextNet2.block_2d(self, channels, channels),
                nn.MaxPool2d(2, 2),
                *CextNet2.block_2d(self, channels, channels),
                layer.MultiStepDropout(0.5),
                *CextNet2.block_2d(self, channels, 110),
                layer.MultiStepDropout(0.5),
                nn.Unflatten(0, (T, b))
            )
            self.vote = VotingLayer(10)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)
            x = self.conv2d(x)
            x = x.permute(1, 2, 0, 3, 4)
            out_spikes = x.flatten(2).permute(2, 0, 1)
            return self.vote(out_spikes.mean(0))

        @staticmethod
        def block_2d(self, in_channels: int, out_channels: int):
            return [
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Unflatten(0, (self.T, self.b)),
                neuron.MultiStepLIFNode(
                    tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
                nn.Flatten(0, 1)
            ]


except ImportError:
    print('Cupy is not installed.')
