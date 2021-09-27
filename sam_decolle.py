import argparse
from typing import Tuple

import math
from utils.neural_coding import neural_coding, rate_coding
from utils.localization_utils import image_to_spikes, iou_metric
from utils.oxford_iiit_pet_loader import OxfordPetDatasetLocalization, get_transforms, init_oxford_dataset

from torch.utils.data.dataloader import DataLoader
from utils.misc import cross_entropy_one_hot, onehot_np, save_prediction_errors, tonp
from snn.base import DECOLLEBase, DECOLLELoss
from models.decolle_cnn import LenetDECOLLE
import os
import random
import shutil
import time
from utils.meters import AverageMeter, ProgressMeter, TensorboardMeter
from utils.args import get_args
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary

# params to change
BATCH_NUMBER = 0
# steepness of the exponential kernel function (in the Spiking Activation Map formula)
GAMMA = 0.4

# GPU if available (or CPU instead)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create the experiment dir if it does not exist
    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.mkdir(os.path.join('experiments', args.experiment))

    # create the sam_imgs dir inside the experiment
    SAM_DIR = os.path.join('experiments', args.experiments, 'sam_imgs')
    if not os.path.exists(SAM_DIR):
        os.mkdir(SAM_DIR)

    model = LenetDECOLLE(
        (1, args.height, args.width),
        Nhid=[32, 64, 128],
        Mhid=[],
        out_channels=4,
        kernel_size=[7],
        stride=[1],
        pool_size=[2, 1, 2],
        alpha=[0.97],
        beta=[0.85],
        num_conv_layers=3,
        num_mlp_layers=0,
        deltat=1000
    )
    model.to(device)

    # define loss function
    loss_fn = [nn.SmoothL1Loss() for _ in range(len(model))]
    if model.with_output_layer:
        loss_fn[-1] = cross_entropy_one_hot

    criterion = DECOLLELoss(net=model, loss_fn=loss_fn)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc'].to(device)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # dataloaders code
    print('Create data loaders')
    val_loader = get_dataloader(args)

    # Initialize parameters
    print('Init parameters')
    data_batch, _, _ = next(iter(val_loader))
    data_batch = neural_coding(data_batch, args)
    data_batch = data_batch.to(device)
    model.init_parameters(data_batch)

    # Create the SAM images
    with torch.no_grad():
        model.eval()

        images, class_id, bbox = val_loader[BATCH_NUMBER]

        images = neural_coding(images, args)
        class_id = onehot_np(class_id, n_classes=2)

        images = images.to(device)
        class_id = torch.Tensor(class_id).to(device)
        bbox = torch.Tensor(bbox).to(device)

        # compute output
        total_loss, layers_miou, layers_act = snn_inference(
            images, bbox, model, criterion, args)

        print('Saving images of batch')  # TODO : make good prints


def snn_inference(images, bbox, model: DECOLLEBase, criterion: DECOLLELoss, args, batch_number=0):
    # burnin phase
    model.init(images.transpose(0, 1), args.burnin)
    t_sample = images.shape[0]

    # per-layer losses for the whole batch
    loss_tv = torch.tensor(0.).to(device)

    # layers activities
    layers_act = [0. for _ in range(len(model))]  # one entry per layer

    # total loss for the whole inference process
    total_loss = torch.tensor(0.).to(device)

    batch_size = images.shape[1]

    # cumulates the predictions for each timestep
    r_cum = np.zeros((len(model), batch_size, args.timesteps - args.burnin, 4))

    # Cumulates the spiking feature maps for each timesteps
    s_cum = [[] for _ in range(len(model))]

    # FOR EACH TIMESTEP
    for k in (range(args.burnin, t_sample)):
        s, r, u = model(images[k, :, :, :])
        loss_ = criterion(
            s, r, u, target=bbox,  sum_=False)

        loss_tv += sum(loss_)
        total_loss += loss_tv

        # update the cumulator of predictions
        r_np = np.array(tonp(r))
        r_cum[:, :, k - args.burnin, :] += r_np

        # update the cumulator of spikes
        for i in range(len(model)):
            s_cum[i].append(s[i])

        for i in range(len(model)):
            layers_act[i] += tonp(s[i].mean().data)/t_sample

        # reinitialize loss_tv
        loss_tv = torch.tensor(0.).to(device)

    # Compute the IoU for each layer
    layers_iou = []
    for i in range(len(model)):
        layers_iou.append(iou_metric(
            r_np[i], bbox.cpu().detach().numpy(), batch_size, args.height, args.width))  # last prediction

        save_prediction_errors(r_cum[i, :, :, :], bbox.cpu().detach().numpy(
        ), args, result_file=f'result_preds_layer{i}_batch{batch_number}.png')

    # Compute the SAM for each layer and each timesteps
    for i in range(len(model)):
        for t in range(args.burnin + 1, t_sample):
            NCS = torch.zeros_like(s_cum[i][0])
            for t_p in range(args.burnin, t):
                mask = s_cum[i][t_p] == 1.
                # formula (12) in the paper of SAM
                NCS[mask] += math.exp(-GAMMA * abs(t - t_p))

            M = torch.sum(NCS, dim=1)
            print(M.shape)

            # resize to full size

    return total_loss, layers_iou, layers_act

def save_heatmaps(M, args):
    for i in range(M.shape[0]):
        heatmap = M[i]
        

def get_dataloader(args) -> DataLoader:
    # Initialize Oxford-IIIT-Pet dataset in data/oxford_iiit_pet directory
    (
        train_images_filenames,
        val_images_filenames,
        images_directory,
        masks_directory,
        xml_directory,
    ) = init_oxford_dataset("data/oxford_augmented")
    # Transforms
    # transform_train = get_transforms(args.height, args.width, is_training=True)
    transform_val = get_transforms(args.height, args.width, is_training=False)

    # Datasets
    # train_data = OxfordPetDatasetLocalization(
    #     train_images_filenames,
    #     images_directory,
    #     masks_directory,
    #     transform=transform_train,
    #     use_DOG=False,  # TODO
    # )

    val_data = OxfordPetDatasetLocalization(
        val_images_filenames,
        images_directory,
        masks_directory,
        transform=transform_val,
        use_DOG=False,
    )

    # Dataloaders
    # must_shuffle = False if args.debug else True
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=must_shuffle, num_workers=args.workers
    # )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    return val_loader


if __name__ == '__main__':
    main()
