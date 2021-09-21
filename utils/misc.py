from collections import Counter
import os
from utils.localization_utils import iou_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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


def save_prediction_errors(preds: np.ndarray, bbox: np.ndarray, args, result_file="prediction_errors.png"):
    """Saves a chart of the prediction errors r_cum compared to the bbox ground truth

    Args:
        r_cum (np.ndarray): list of the predictions for each timesteps. Shape=(Batch, Timesteps, 4)
        bbox (np.ndarray): the ground truth bounding box. Shape=(Batch, 4)
        result_file (str, optional): the filename of the image that will be saved. Defaults to "prediction_errors.png".
    """
    timesteps = preds.shape[1]
    ious = []
    for t in range(timesteps):
        iou = iou_metric(preds[:, t, :], bbox,
                         bbox.shape[0], args.height, args.width)
        ious.append(iou)

    plt.plot(range(args.burnin, args.timesteps), ious, marker='o')
    plt.title('IoU of prediction per timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('IoUs of predictions')
    plt.grid(True)
    plt.savefig(result_file)
    plt.close()
