import argparse
import os
import random
import shutil
import time
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
from utils.localization_utils import iou_metric
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
from utils.oxford_iiit_pet_loader import OxfordPetDatasetLocalization, get_transforms, init_oxford_dataset

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt


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
    model = ResNet5(2, 4, args.timesteps)

    # TODO: define loss function
    criterion = nn.CrossEntropyLoss().to(device)

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
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc, loss = one_epoch(train_loader, model, criterion,
                              epoch, args, tensorboard_meter, optimizer=optimizer)  # TODO

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


def one_epoch(dataloader, model, criterion, epoch, args, tensorboard_meter: TensorboardMeter = None, optimizer=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode. 
    """
    # TODO: define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ious = AverageMeter('IoU', ':6.2f')

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

        images = images.to(device)

        bbox_pred = model(images)

        bbox_pred = torch.clip(bbox_pred, 0., 1.)  # clips value

        loss = criterion(bbox_pred, bbox)

        # measure accuracy and record loss
        # TODO: define accuracy metrics
        iou = iou_metric(bbox_pred, bbox, args.batch_size,
                         args.height, args.width)
        losses.update(loss.item(), images.size(0))
        ious.update(iou, images.size(0))

        # compute gradient and do SGD step (if training)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # if debugging, stop after the first batch
        if args.debug:
            break

        # TODO: define AverageMeters used in tensorboard summary
        if is_training:
            tensorboard_meter.update_train([ious, losses])
        else:
            tensorboard_meter.update_val([ious, losses], epoch)

    return ious.avg, losses.avg  # TODO


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
        transform=transform_train,
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
