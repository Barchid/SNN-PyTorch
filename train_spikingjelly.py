import argparse
import os
import random
import shutil
import time

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from utils.meters import AverageMeter, ProgressMeter, TensorboardMeter
from utils.args import get_args
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda import amp
from torchsummary import summary

from spikingjelly.clock_driven import functional

# GPU if available (or CPU instead)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: best accuracy metrics (used to save the best checkpoints)
best_acc = 0.


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

    # TODO: define model
    model = None

    # TODO: define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # TODO: define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # TODO: define input_size here to have the right summary of your model
    if args.summary:
        summary(model, input_size=(3, 224, 224))
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
    train_dataset = DVS128Gesture(
        args.data, train=True, data_type='frame', split_by='number', frames_number=args.timesteps)
    val_dataset = DVS128Gesture(args.data, train=False,
                                data_type='frame', split_by='number', frames_number=args.timesteps)

    must_shuffle = False if args.debug else True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=must_shuffle, num_workers=args.workers, pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True
    )

    # If only evaluating the model is required
    if args.evaluate:
        with torch.no_grad():
            _, _, _ = one_epoch(val_loader, model, criterion,
                                0, args, optimizer=None)
        return

    # define tensorboard meter
    tensorboard_meter = TensorboardMeter(f"experiments/{args.experiment}/logs")

    # Activate grad scaler if mixed precision is enabled
    scaler = None
    if args.mixed_precision:
        scaler = amp.GradScaler()

    # TRAINING + VALIDATION LOOP
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc, loss = one_epoch(train_loader, model, criterion,
                              epoch, args, tensorboard_meter, scaler=scaler, optimizer=optimizer)  # TODO

        # evaluate on validation set (optimizer is None when validation)
        with torch.no_grad():
            acc, loss = one_epoch(
                val_loader, model, criterion, epoch, args, tensorboard_meter, scaler=None, optimizer=None)

        # remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'{args.experiment}/checkpoint_{str(epoch).zfill(5)}.pth.tar')


def one_epoch(dataloader, model, criterion, epoch, args, tensorboard_meter: TensorboardMeter = None, scaler=None, optimizer=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode. 
    """
    # TODO: define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Accuracy', ':6.2f')

    is_training = optimizer is not None
    prefix = 'TRAIN' if is_training else 'TEST'

    # TODO: final Progress Meter (add the relevant AverageMeters)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, accuracies],
        prefix=f"{prefix} - Epoch: [{epoch}]")

    # switch to train mode (if training)
    if is_training:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.float().to(device)
        target = target.to(device)
        target_onehot = F.one_hot(target, 11)

        # compute output
        if scaler is None:  # full precision
            output = model(images)
            loss = criterion(output, target_onehot)
        else:  # automatic mixed precision
            with amp.autocast():
                output = model(images)
                loss = criterion(output, target_onehot)

        # measure accuracy and record loss
        accuracy = (output.argmax(dim=1) == target).float().sum().item()
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy[0], images.size(0))

        # compute gradient and do SGD step (if training)
        if is_training:
            optimizer.zero_grad()
            if scaler is None:  # full precision
                loss.backward()
                optimizer.step()
            else:  # automatic mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # reset the network's activities
        functional.reset_net(model)

        if i % args.print_freq == 0:
            progress.display(i)

        # if debugging, stop after the first batch
        if args.debug:
            break

        # TODO: define AverageMeters used in tensorboard summary
        if is_training:
            tensorboard_meter.update_train([accuracies, losses])
        else:
            tensorboard_meter.update_val([accuracies, losses])

        return accuracies.avg, losses.avg  # TODO


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
