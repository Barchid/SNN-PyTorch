"""
    Arguments available to train scripts.
"""


import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8).')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1).', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--summary', action='store_true',
                        help="Prints a summary of the model to be trained.")

    # training related
    parser.add_argument('--experiment', required=True,
                        type=str, help="Name of the experiment.")
    parser.add_argument('--debug', action='store_true',
                        help="If used, forces to overfit only one batch of the train split (to debug the network).")

    # SNN online learning related
    parser.add_argument('--timesteps', '-t', type=int, default=1000,
                        help="Delta T.")
    parser.add_argument('--burnin', type=int, default=50,
                        help="Number of steps at the beginning of the forward pass where no learning rule is applied. It is used to put some activity in the SNN (default: 50).")

    parser.add_argument('--mixed-precision', action="store_true",
                        help="Enables the Automatic Mixed Precision optimization from PyTorch.")

    parser.add_argument('--height', type=int,
                        help="Height dimension of the input", default=176)
    parser.add_argument('--width', type=int,
                        help="Width dimension of the input", default=240)

    # SNN coding of static image
    parser.add_argument('--neural-coding', type=str,
                        choices=['rate', 'ttfs', 'phase', 'burst'], default='rate')

    # Parameter to save some metrics
    parser.add_argument('--save-preds', type=int, default=0,
                        help="Batch number where metrics will be saved.")
    return parser.parse_args()
