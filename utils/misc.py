import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def tonp(tensor):
    if type(tensor) == type([]):
        return [t.detach().cpu().numpy() for t in tensor]
    elif not hasattr(tensor, 'detach'):
        return tensor
    else:
        return tensor.detach().cpu().numpy()
