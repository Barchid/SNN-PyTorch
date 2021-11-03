import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .localization_utils import iou_metric


class DIoULoss(nn.Module):
    """Distance-IoU loss function for single object localization"""

    def __init__(self):
        super(DIoULoss, self).__init__()

    def forward(self, pred, gt):
        x1 = pred[:, 0]
        y1 = pred[:, 1]
        x2 = pred[:, 2]
        y2 = pred[:, 3]

        x1g = gt[:, 0]
        y1g = gt[:, 1]
        x2g = gt[:, 2]
        y2g = gt[:, 3]

        x2 = torch.max(x1, x2)
        y2 = torch.max(y1, y2)

        x_p = (x2 + x1) / 2
        y_p = (y2 + y1) / 2
        x_g = (x1g + x2g) / 2
        y_g = (y1g + y2g) / 2

        xkis1 = torch.max(x1, x1g)
        ykis1 = torch.max(y1, y1g)
        xkis2 = torch.min(x2, x2g)
        ykis2 = torch.min(y2, y2g)

        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        intsctk = torch.zeros(x1.size()).to(pred)
        mask = (ykis2 > ykis1) * (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * \
            (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * \
            (y2g - y1g) - intsctk + 1e-7
        iouk = intsctk / unionk

        c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
        d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
        u = d / c
        diouk = iouk - u
        diouk = (1 - diouk).sum(0) / pred.size(0)

        return diouk


def compute_IoU(pred, gt):
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = pred[:, 2]
    y2 = pred[:, 3]

    x1g = gt[:, 0]
    y1g = gt[:, 1]
    x2g = gt[:, 2]
    y2g = gt[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(pred)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * \
        (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * \
        (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    iouk = iouk.sum(0) / pred.size(0)

    return iouk

# def compute_diou(pred, gt):
#     x1 = pred[:, 0]
#     y1 = pred[:, 1]
#     x2 = pred[:, 2]
#     y2 = pred[:, 3]

#     x1g = gt[:, 0]
#     y1g = gt[:, 1]
#     x2g = gt[:, 2]
#     y2g = gt[:, 3]

#     x2 = torch.max(x1, x2)
#     y2 = torch.max(y1, y2)

#     x_p = (x2 + x1) / 2
#     y_p = (y2 + y1) / 2
#     x_g = (x1g + x2g) / 2
#     y_g = (y1g + y2g) / 2

#     xkis1 = torch.max(x1, x1g)
#     ykis1 = torch.max(y1, y1g)
#     xkis2 = torch.min(x2, x2g)
#     ykis2 = torch.min(y2, y2g)

#     xc1 = torch.min(x1, x1g)
#     yc1 = torch.min(y1, y1g)
#     xc2 = torch.max(x2, x2g)
#     yc2 = torch.max(y2, y2g)

#     intsctk = torch.zeros(x1.size()).to(pred)
#     mask = (ykis2 > ykis1) * (xkis2 > xkis1)
#     intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
#     unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
#     iouk = intsctk / unionk

#     c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
#     d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
#     u = d / c
#     diouk = iouk - u
#     iouk = iouk.sum(0) / pred.size(0)
#     diouk = (1 - diouk).sum(0) / pred.size(0)

#     return iouk, diouk
