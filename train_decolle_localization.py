import argparse
from typing import Tuple
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
import math
import cv2

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
import matplotlib.pyplot as plt
from celluloid import Camera

# GPU if available (or CPU instead)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SAM hyperparameter
GAMMA = 0.4


def main():
    # TODO: best accuracy metrics (used to save the best checkpoints)
    best_acc = 0.

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

    # define optimizer
    optimizer = torch.optim.Adamax(
        model.get_trainable_parameters(),
        args.lr,
        betas=[0.0, 0.95]
    )

    # define input_size here to have the right summary of your model
    if args.summary:
        summary(model, input_size=(1, args.height, args.width))
        exit()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc'].to(device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # dataloaders code
    print('Create data loaders')
    train_loader, val_loader = get_dataloaders(args)

    # Initialize parameters
    print('Init parameters')
    data_batch, _, _ = next(iter(train_loader))
    data_batch = neural_coding(data_batch, args)
    data_batch = data_batch.to(device)
    model.init_parameters(data_batch)

    # If only evaluating the model is required
    if args.evaluate:
        _, _, _ = one_epoch(val_loader, model, criterion,
                            0, args, optimizer=None)
        return

    # define tensorboard meter
    tensorboard_meter = TensorboardMeter(f"experiments/{args.experiment}/logs")

    # TRAINING + VALIDATION LOOP
    print('BEGINNING TRAINING LOOP')
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        accs, loss = one_epoch(train_loader, model, criterion,
                               epoch, args, tensorboard_meter, optimizer=optimizer)

        # evaluate on validation set (optimizer is None when validation)
        if not args.debug:
            accs, loss = one_epoch(
                val_loader, model, criterion, epoch, args, tensorboard_meter, optimizer=None)

        acc = accs[-1]  # accuracy of last layer

        # remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'experiments/{args.experiment}/checkpoint_{str(epoch).zfill(5)}.pth.tar')

    tensorboard_meter.close()


def one_epoch(dataloader, model, criterion, epoch, args, tensorboard_meter: TensorboardMeter, optimizer=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode.
    """
    # define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mious = [AverageMeter(
        f'mIoU of layer {i}', ':6.2f') for i in range(len(model))]
    activities = [AverageMeter(
        f'Activity rate of layer {i}', ':6.2f') for i in range(len(model))]

    is_training = optimizer is not None
    prefix = 'TRAIN' if is_training else 'TEST'

    # final Progress Meter (add the relevant AverageMeters)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, *mious, *activities],
        prefix=f"{prefix} - Epoch: [{epoch}]")

    # switch to train mode (if training)
    if is_training:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (images, class_id, bbox) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        grayscales = images.numpy()
        images = neural_coding(images, args)
        class_id = onehot_np(class_id, n_classes=2)

        images = images.to(device)
        class_id = torch.Tensor(class_id).to(device)
        bbox = torch.Tensor(bbox).to(device)

        # compute output
        total_loss, layers_miou, layers_act = snn_inference(
            images, bbox, model, criterion, optimizer, args, is_training, grayscales, batch_number=i)

        # measure accuracy and record loss
        losses.update(total_loss.item(), images.size(0))
        for j, miou in enumerate(mious):
            miou.update(layers_miou[j], images.size(0))
        for j, activity in enumerate(activities):
            activity.update(layers_act[j], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or args.debug:
            progress.display(i)

        # if debugging, stop after the first batch
        if args.debug:
            break

    # TODO: define AverageMeters used in tensorboard summary
    if is_training:
        tensorboard_meter.update_train([*mious, *activities, losses])
    else:
        tensorboard_meter.update_val(
            [*mious, *activities, losses], epoch=epoch)

    return [miou.avg for miou in mious], losses.avg  # TODO


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def snn_inference(images, bbox, model: DECOLLEBase, criterion: DECOLLELoss, optimizer, args, is_training, grayscales, batch_number=0):
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

        # ONLINE LEARNING UPDATE
        if is_training:
            loss_tv.backward()
            optimizer.step()
            optimizer.zero_grad()

        # update the cumulator of predictions
        r_np = np.array(tonp(r))
        r_cum[:, :, k - args.burnin, :] += r_np

        # update the cumulator of spikes if visualization used
        if batch_number % args.save_preds == 0:
            for i in range(len(model)-1):
                s_cum[i].append(s[i].detach().cpu())
            s_cum[-1].append((u[-1].detach().cpu() >= 0.).float())

        for i in range(len(model)):
            layers_act[i] += tonp(s[i].mean().data)/t_sample

        # reinitialize loss_tv
        loss_tv = torch.tensor(0.).to(device)

    # Compute the IoU for each layer
    layers_iou = []
    for i in range(len(model)):
        layers_iou.append(iou_metric(
            r_np[i], bbox.cpu().detach().numpy(), batch_size, args.height, args.width))  # last prediction

        if batch_number % args.save_preds == 0:
            prefix = "TRAIN" if is_training else "TEST"
            save_prediction_errors(
                r_cum[i, :, :, :], bbox.cpu().detach().numpy(), args, result_file=f'{args.experiment}_{prefix}_result_preds_layer{i}_batch{str(batch_number).zfill(5)}.png')

            # Compute the SAM for each layer and each timesteps
            for t in range(args.burnin + 1, t_sample):
                heatmaps = []
                NCS = torch.zeros_like(s_cum[i][0])
                for t_p in range(args.burnin, t):
                    mask = s_cum[i][t_p] == 1.
                    # formula (12) in the paper of SAM
                    NCS[mask] += math.exp(-GAMMA * abs(t - t_p))

                M = torch.sum(NCS * s_cum[i][t], dim=1)
                heatmap = create_heatmap(
                    M, grayscales, i, t, args, batch_number, prefix)

                heatmaps.append(heatmap)

                # save video of the SAM evolution when we arrive at the last timestep
                if t == t_sample - 1:
                    heatmap_video(
                        heatmaps, filename=f"{args.experiment}/{prefix}_L{i}_B{str(batch_number).zfill(4)}.mp4")

    return total_loss, layers_iou, layers_act


def get_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    # Initialize Oxford-IIIT-Pet dataset in data/oxford_iiit_pet directory
    (
        train_images_filenames,
        val_images_filenames,
        images_directory,
        masks_directory,
        xml_directory,
    ) = init_oxford_dataset("data/oxford_augmented")
    # Transforms
    transform_train = get_transforms(args.height, args.width, is_training=True)
    transform_val = get_transforms(args.height, args.width, is_training=False)

    # Datasets
    train_data = OxfordPetDatasetLocalization(
        train_images_filenames,
        images_directory,
        masks_directory,
        transform=transform_train,
        use_DOG=False,  # TODO
    )

    val_data = OxfordPetDatasetLocalization(
        val_images_filenames,
        images_directory,
        masks_directory,
        transform=transform_val,
        use_DOG=False,
    )

    # Dataloaders
    must_shuffle = False if args.debug else True
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=must_shuffle, num_workers=args.workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    return train_loader, val_loader


def create_heatmap(M, images, layer, timestep, args, batch_number, prefix):
    for i in range(M.shape[0]):
        heatmap = M[i].numpy()

        # normalize between 0 and 1
        max = np.max(heatmap)
        min = np.min(heatmap)
        heatmap = (heatmap - min) / (max - min)

        # resize the heatmap
        heatmap = cv2.resize(heatmap, (args.width, args.height))

        # three channeled version of the grayscale image
        image = cv2.cvtColor(images[i][0], cv2.COLOR_GRAY2RGB)

        heatmap = show_cam_on_image(image, heatmap, use_rgb=True)

        # save the heatmap
        print(
            f'Saving SAM of:Batch={batch_number}\t\tImage={i}\t\tLayer={layer}\t\tTimestep={timestep}')
        cv2.imwrite(
            f'SAM/{args.experiment}_{prefix}_batch{str(batch_number).zfill(5)}_image{i}_l{layer}_ts{str(timestep).zfill(4)}.png', heatmap)

        return heatmap


def heatmap_video(heatmaps, filename):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    plt.axis("off")

    for heatmap in heatmaps:
        ax.imshow(heatmap)
        camera.snap()

    anim = camera.animate(interval=40)
    anim.save(filename)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    main()
