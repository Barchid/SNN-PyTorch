import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Energy(object):
    """Computes the energy consumption of the Spike layer in parameters following the method in https://arxiv.org/pdf/2110.07742.pdf"""

    def __init__(self, layer: nn.Module, C_in: int, C_out: int, k: int = 1, O: int = 1):
        super(Energy, self).__init__()

        self.hook = layer.register_forward_hook(self.hook_save_spikes)
        self.neuron_number = None  # initialized at the first hook call
        self.spike_count = 0

    def hook_save_spikes(self, module, input, output):
        spikes = output[0].detach().cpu().numpy()

        if self.neuron_number is None:
            self.neuron_number = spikes.size
            self.spike_count = np.count_nonzero(spikes)
