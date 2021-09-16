from collections import Counter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def tonp(tensor):
    if type(tensor) == type([]):
        return [t.detach().cpu().numpy() for t in tensor]
    elif not hasattr(tensor, 'detach'):
        return tensor
    else:
        return tensor.detach().cpu().numpy()


def onehot_np(tensor: np.ndarray, n_classes: int):
    return np.eye(n_classes)[tensor]


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return torch.nn.CrossEntropyLoss()(input, labels)


def prediction_mostcommon(outputs):
    maxs = outputs.argmax(axis=-1)
    res = []
    for m in maxs:
        most_common_out = []
        for i in range(m.shape[1]):
            #            idx = m[:,i]!=target.shape[-1] #This is to prevent classifying the silence states
            most_common_out.append(Counter(m[:, i]).most_common(1)[0][0])
        res.append(most_common_out)
    return res


def accuracy(outputs, targets, one_hot=True):
    if type(targets) is torch.Tensor:
        targets = tonp(targets)

    return [np.mean(o == targets) for o in outputs]
