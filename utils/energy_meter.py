import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class EnergyMeter(object):
    """Computes the energy consumption of the Spike layer in parameters following the method in https://arxiv.org/pdf/2110.07742.pdf"""

    def __init__(self, layer: nn.Module, C_in: int, C_out: int, k: int = 1, O: int = 1):
        super(EnergyMeter, self).__init__()

        self.hook = layer.register_forward_hook(self.hook_save_spikes)
        self.neuron_number = None  # initialized at the first hook call
        self.spike_count = 0
        self.C_in = float(C_in)
        self.C_out = float(C_out)
        self.k = float(k)
        self.O = float(O)

    def hook_save_spikes(self, module, input, output):
        spikes = output[0].detach().cpu().numpy()
        if self.neuron_number is None:
            self.neuron_number = spikes.size
            self.spike_count = np.count_nonzero(spikes)
        else:
            self.spike_count = self.spike_count + np.count_nonzero(spikes)

    def get_energy_consumption(self):
        spike_rate = self.spike_count / self.neuron_number

        flops_ann = (self.k**2) * (self.O**2) * self.C_in * self.C_out

        flops_snn = flops_ann * spike_rate

        # reinitialize after computation
        self.neuron_number = None
        self.spike_count = 0

        return flops_snn, flops_ann
