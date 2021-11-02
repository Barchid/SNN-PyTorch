import argparse
import os
import random
import shutil
import time
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
from utils.SAM_hook import SAM, heatmap_video
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
from utils.neural_coding import neural_coding
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


def get_SAM(model: ResNet9, args):
    spike1 = SAM(model.spike1, args.height, args.width)
    res2_spike1 = SAM(model.res2_spike1, args.height, args.width)
    res2_spike2 = SAM(model.res2_spike2, args.height, args.width)
    res3_spike1 = SAM(model.res3_spike1, args.height, args.width)
    res3_spike2 = SAM(model.res3_spike2, args.height, args.width)
    res4_spike1 = SAM(model.res4_spike1, args.height, args.width)
    res4_spike2 = SAM(model.res4_spike2, args.height, args.width)
    res5_spike1 = SAM(model.res5_spike1, args.height, args.width)
    res5_spike2 = SAM(model.res5_spike2, args.height, args.width)

    return {
        'spike1': spike1,
        'res2_spike1': res2_spike1,
        'res2_spike2': res2_spike2,
        'res3_spike1': res3_spike1,
        'res3_spike2': res3_spike2,
        'res4_spike1': res4_spike1,
        'res4_spike2': res4_spike2,
        'res5_spike1': res5_spike1,
        'res5_spike2': res5_spike2
    }


def main():
    # TODO: best accuracy metrics (used to save the best checkpoints)
    best_acc = 0.

    args = get_args()

    # batch size is 1 for testing
    args.batch_size = 1

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
    criterion = nn.SmoothL1Loss().to(device)

    # TODO: define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999)
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # TODO: dataloaders code
    train_loader, val_loader = get_dataloaders(args)

    data_loader = train_loader if args.debug else val_loader

    with torch.no_grad():
        _, _, _ = one_epoch(val_loader, model, criterion,
                            0, args, sams=get_SAM(model, args))


def one_epoch(dataloader, model, criterion, epoch, args, sams={}):
    """One epoch pass. If the optimizer is not None, the function works in training mode. 
    """
    # TODO: define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.7f')
    ious = AverageMeter('IoU', ':6.3f')

    prefix = 'TEST'

    # TODO: final Progress Meter (add the relevant AverageMeters)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, ious],
        prefix=f"{prefix} - Epoch: [{epoch}]")

    # switch to train mode (if training)
    model.eval()

    end = time.time()
    for i, (images, class_id, bbox) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        neural_images = neural_coding(images, args)
        neural_images = neural_images.to(device)
        bbox = bbox.to(device)

        bbox_pred = model(neural_images)

        # save the SAMs output of the first image in the batch
        if not os.path.exists(os.path.join('experiments', args.experiment, 'SAMS')):
            os.mkdir(os.path.join('experiments', args.experiment, 'SAMS'))

        for name, sam in sams.items():
            heatmaps = sam.get_sam()
            # take only the heatmap of the first image in the batch
            heatmaps = [hm[0] for hm in heatmaps]
            heatmap_video(images.detach().cpu().numpy()[0], heatmaps, os.path.join(
                'experiments', args.experiment, 'SAMS', f"{name}____b{i:6.1f}.mp4"))

        loss = criterion(bbox_pred, bbox)

        # measure accuracy and record loss
        iou = iou_metric(bbox_pred.detach().cpu().numpy(), bbox.detach().cpu().numpy(), images.size(0),
                         args.height, args.width)
        losses.update(loss.item(), images.size(0))
        ious.update(iou, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

        if args.debug:
            print(f'\nPRED={bbox_pred[0]}\nGT={bbox[0]}')
            return iou, loss

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
