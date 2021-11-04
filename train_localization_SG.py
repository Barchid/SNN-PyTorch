import argparse
import os
import random
import shutil
import time
from typing import Tuple
from torch.nn.functional import smooth_l1_loss

from torch.utils.data.dataloader import DataLoader
from models.snntorch_baseline import Baseline5
from utils.diou_loss import DIoULoss, compute_IoU
from utils.localization_utils import draw_bbox, format_bbox, iou
from utils.meters import AverageMeter, ProgressMeter, TensorboardMeter
from utils.args_snntorch import get_args
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda import amp
from torchsummary import summary
from models.snntorch_sewresnet import ResNet5, ResNet9
from utils.neural_coding import neural_coding
from utils.oxford_iiit_pet_loader import OxfordPetDatasetLocalization, get_transforms, init_oxford_dataset

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt


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

    # TODO: define model
    model = ResNet9(
        # if on/off filtering, there is 2 channels (else, there is 1)
        2 if args.on_off else 1,
        4,
        args.timesteps
    ).to(device)

    # TODO: define loss function
    # criterion = nn.SmoothL1Loss().to(device)
    criterion = nn.SmoothL1Loss().to(device)

    # TODO: define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999)
    )

    # TODO: define input_size here to have the right summary of your model
    if args.summary:
        summary(model, input_size=(2, 28, 28))
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

    # TODO: dataloaders code
    train_loader, val_loader = get_dataloaders(args)

    # If only evaluating the model is required
    if args.evaluate:
        with torch.no_grad():
            _, _, _ = one_epoch(val_loader, model, criterion,
                                0, args, optimizer=None)
        return

    # define tensorboard meter
    tensorboard_meter = TensorboardMeter(f"experiments/{args.experiment}/logs")

    # TRAINING + VALIDATION LOOP
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc, loss = one_epoch(train_loader, model, criterion,
                              epoch, args, tensorboard_meter, optimizer=optimizer)  # TODO

        if args.debug:
            best_acc = acc
            continue

        # evaluate on validation set (optimizer is None when validation)
        with torch.no_grad():
            acc, loss = one_epoch(
                val_loader, model, criterion, epoch, args, tensorboard_meter, optimizer=None)

        # remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'experiments/{args.experiment}/checkpoint_{str(epoch).zfill(5)}.pth.tar')

    # save the debugged model at the end of the script
    if args.debug:
        save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=f'experiments/{args.experiment}/debug.pth.tar')


def one_epoch(dataloader, model, criterion, epoch, args, tensorboard_meter: TensorboardMeter = None, optimizer=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode. 
    """
    # TODO: define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.7f')
    ious = AverageMeter('IoU', ':6.3f')

    is_training = optimizer is not None
    prefix = 'TRAIN' if is_training else 'TEST'

    # TODO: final Progress Meter (add the relevant AverageMeters)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, ious],
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

        neural_images = neural_coding(images, args)
        neural_images = neural_images.to(device)
        bbox = bbox.to(device)

        bbox_preds = model(neural_images)

        # loss = criterion(bbox_pred, bbox)
        final_loss = torch.zeros((1), device=device)
        for bbox_pred in bbox_preds:
            loss = criterion(bbox_pred, bbox)
            final_loss += loss

        # compute gradient and do SGD step (if training)
        if is_training:
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        iou = compute_IoU(bbox_pred.detach().cpu(), bbox.detach().cpu())
        losses.update(loss.item(), images.size(0))
        ious.update(iou.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # if debugging, stop after the first batch
        if args.debug:
            save_sample(bbox_pred, bbox, images, epoch, args)
            break

        # save sample if required (only in test phase)
        if not is_training and i == args.save_sample:
            save_sample(bbox_pred, bbox, images, epoch, args)

        # TODO: define AverageMeters used in tensorboard summary
        if is_training:
            tensorboard_meter.update_train([ious, losses])
        else:
            tensorboard_meter.update_val([ious, losses], epoch)

    return ious.avg, losses.avg  # TODO


def save_sample(bbox_pred, bbox_gt, image, epoch, args):
    # always the first image in the batch
    image = image.detach().cpu().numpy()[0]
    bbox_pred = bbox_pred.detach().cpu().numpy()[0]
    bbox_gt = bbox_gt.detach().cpu().numpy()[0]

    parent_dir = os.path.join('experiments', args.experiment, 'samples')
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    # IoU computation
    pred = format_bbox(bbox_pred, args.height, args.width)
    gt = format_bbox(bbox_gt, args.height, args.width)
    IoU = iou(pred, gt)

    filename = os.path.join(
        parent_dir, f"ep{str(epoch).zfill(4)}_iou{IoU:3.2f}.png")

    # draw bbox on the image
    image = image[0]  # get rid of useless channel
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im_pred = draw_bbox(image, pred)
    im_gt = draw_bbox(image, gt)
    title = f"IoU = {IoU}"
    fig.suptitle(title)
    ax1.imshow(im_pred)
    ax2.imshow(im_gt)
    plt.savefig(filename)
    plt.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        transform=transform_train if not args.debug else transform_val,
        use_DOG=args.on_off,  # TODO
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


if __name__ == '__main__':
    main()
