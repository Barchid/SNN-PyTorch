from utils.snn_utils import spiketrains
import torch
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import argparse
from . import oxford_iiit_pet_loader as IIT
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def image_to_spikes(
    input,
    gain=50,
    min_duration=None,
    max_duration=1000,
    input_shape=(1, 28, 28),
):
    """
    Transforms the input image (in params) into spike trains and the corresponding
    segmentation mask into a one-hot encoded version (no spike trains)
    """
    batch_size = input.shape[0]

    if min_duration is None:
        min_duration = max_duration - 1

    # parsing input
    T = np.random.randint(min_duration, max_duration, batch_size)
    Nin = np.prod(input_shape)
    allinputs = np.zeros([batch_size, max_duration, Nin])
    for i in range(batch_size):
        st = spiketrains(T=T[i], N=Nin, rates=gain * input[i].reshape(-1)).astype(
            np.float32
        )
        allinputs[i] = np.pad(st, ((0, max_duration - T[i]), (0, 0)), "constant")
    allinputs = np.transpose(allinputs, (1, 0, 2))
    allinputs = allinputs.reshape(
        allinputs.shape[0],
        allinputs.shape[1],
        input_shape[0],
        input_shape[1],
        input_shape[2],
    )

    return torch.FloatTensor(allinputs)


def display_predictions(
    images, class_gts, class_preds, bbox_gts, bbox_preds, HEIGHT, WIDTH, is_save = False, filename=None
):
    # from one-hot encoding to argmax encoding
    class_preds = class_preds.clone().detach().argmax(1).cpu().numpy()
    class_gts = class_gts.argmax(1).cpu().numpy()
    bbox_gts = bbox_gts.cpu().numpy()
    bbox_preds = np.clip(
        bbox_preds.clone().detach().cpu().numpy(), a_min=0.0, a_max=1.0
    )
    images = images.clone().detach().cpu().numpy()

    # for each prediction in batch
    for image, class_gt, class_pred, bbox_gt, bbox_pred in zip(
        images, class_gts, class_preds, bbox_gts, bbox_preds
    ):
        image = image[0] # get rid of useless channel
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # From [0,1] dimensions to [0,WIDTH] & [0, HEIGHT]
        bbox_pred = format_bbox(bbox_pred, HEIGHT, WIDTH)
        bbox_gt = format_bbox(bbox_gt, HEIGHT, WIDTH)

        print(f"BBOX GT={bbox_gt}\t\t\tBBOX PRED={bbox_pred}")
        print("IOU =", iou(bbox_gt, bbox_pred))

        # draw bboxes on image
        im_pred = draw_bbox(image, bbox_pred)
        im_gt = draw_bbox(image, bbox_gt)

        title = f"Class_Pred={class_pred}      Class_GT={class_gt}"

        fig.suptitle(title)
        ax1.imshow(im_pred)
        ax2.imshow(im_gt)
        if is_save:
            plt.savefig(f'{filename}_IOU_{iou(bbox_gt, bbox_pred):4f}.png')
        else:
            plt.show()
        plt.close()


def format_bbox(bbox, HEIGHT, WIDTH):
    bbox[0] = int(bbox[0] * WIDTH)  # x min
    bbox[1] = int(bbox[1] * HEIGHT)  # y min
    bbox[2] = int(bbox[2] * WIDTH)  # x max
    bbox[3] = int(bbox[3] * HEIGHT)  # y max
    return bbox


def draw_bbox(image, bbox):
    im = np.copy(image)
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    return cv2.rectangle(
        im, (x_min, y_min), (x_max, y_max), color=(0.5, 0.5, 0.5), thickness=4
    )


def iou(boxA, boxB):
    """
    boxA -- first box, list object with coordinates (x1, y1, x2, y2)
    boxB -- second box, list object with coordinates (x1, y1, x2, y2)
    Source from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def iou_metric(preds, gts, batch_size, HEIGHT, WIDTH):
    iou_sum = 0.0

    for pred, gt in zip(preds, gts):
        pred = format_bbox(pred, HEIGHT, WIDTH)
        gt = format_bbox(gt, HEIGHT, WIDTH)
        print(pred, gt)
        iou_sum += iou(pred, gt)

    return iou_sum / float(batch_size)