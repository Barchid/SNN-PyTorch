import argparse
from utils.misc import tonp
from torchneuromorphic.nmnist.nmnist_dataloaders import create_dataloader
from snn.base import DECOLLEBase, DECOLLELoss
from models.nmnist_cnn import LenetDECOLLE
import os
import random
import shutil
import time
from utils.meters import AverageMeter, ProgressMeter, TensorboardMeter
from utils.args import get_args
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary

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
    if not os.path.exists(os.path.join('experiments', args.experiments)):
        os.mkdir(os.path.join('experiments', args.experiments))

    model = LenetDECOLLE(
        (2, 34, 34),
        Nhid=[64, 128, 128],
        Mhid=[],
        out_channels=10,
        kernel_size=[7],
        stride=[1],
        pool_size=[2, 1, 2],
        alpha=[0.97],
        beta=[0.85],
        num_conv_layers=3,
        num_mlp_layers=0,
        deltat=1000
    )

    # define loss function
    criterion = DECOLLELoss(
        net=model, loss_fn=[nn.SmoothL1Loss() for _ in range(len(model))])

    # define optimizer
    optimizer = torch.optim.Adamax(
        model.get_trainable_parameters(),
        args.lr,
        betas=[0.0, 0.95]
    )

    # define input_size here to have the right summary of your model
    if args.summary:
        summary(model, input_size=(2, 34, 34))
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
    must_shuffle = False if args.debug else True
    train_loader, val_loader = create_dataloader(
        root="data/nmnist/n_mnist.hdf5",
        chunk_size_train=300,
        chunk_size_test=300,
        batch_size=args.batch_size,
        dt=args.timesteps,
        num_workers=args.workers,
        shuffle_train=must_shuffle
    )

    # If only evaluating the model is required
    if args.evaluate:
        _, _, _ = one_epoch(val_loader, model, criterion,
                            0, args, optimizer=None)
        return

    # define tensorboard meter
    tensorboard_meter = TensorboardMeter(f"experiments/{args.experiment}/logs")

    # TRAINING + VALIDATION LOOP
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        acc, loss = one_epoch(train_loader, model, criterion,
                              epoch, args, tensorboard_meter, optimizer=optimizer)

        # evaluate on validation set (optimizer is None when validation)
        acc, loss = one_epoch(val_loader, model, criterion,
                              epoch, args, tensorboard_meter, optimizer=None)

        # remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'{args.experiment}/checkpoint_{str(epoch).zfill(5)}.pth.tar')

    tensorboard_meter.close()


def one_epoch(dataloader, model, criterion, epoch, args, tensorboard_meter: TensorboardMeter, optimizer=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode.
    """
    # define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Accuracy', ':6.2f')

    is_training = optimizer is not None
    prefix = 'TRAIN' if is_training else 'TEST'

    # final Progress Meter (add the relevant AverageMeters)
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

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # TODO: define accuracy metrics
        accuracy = torch.Tensor(1.1)  # TODO: here
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy[0], images.size(0))

        if is_training:
            # compute gradient and do SGD step
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
            tensorboard_meter.update_train([accuracies, losses])
        else:
            tensorboard_meter.update_val([accuracies, losses])

        return accuracies.avg, losses.avg  # TODO


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def snn_inference(images, targets, model: DECOLLEBase, criterion: DECOLLELoss, optimizer, args):
    loss_mask = (targets.sum(2) > 0).unsqueeze(2).float()

    # burnin phase
    model.init(images, args.burnin)
    t_sample = images.shape[1]

    # per-layer losses for the whole batch
    loss_tv = torch.tensor(0.).to(device)

    # total loss for the whole inference process
    total_loss = torch.tensor(0.).to(device)

    # FOR EACH TIMESTEPS
    for k in (range(args.burnin, t_sample)):
        s, r, u = model(images[:, k, :, :])
        loss_ = criterion(
            s, r, u, target=targets[:, k, :], mask=loss_mask[:, k, :], sum_=False)

        loss_tv += sum(loss_)
        total_loss += loss_tv

        # ONLINE LEARNING UPDATE
        loss_tv.backward()
        optimizer.step()
        optimizer.zero_grad()

        # for i in range(len(model)):
            # act_rate[i] += tonp(s[i].mean().data)/t_sample

        # reinitialize loss_tv 
        loss_tv = torch.tensor(0.).to(device)

    return total_loss


if __name__ == '__main__':
    main()
