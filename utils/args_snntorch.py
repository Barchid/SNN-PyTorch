"""
    Arguments available to train scripts.
"""


import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('data', metavar='DIR',
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

    # SNN coding of static image
    parser.add_argument('--neural-coding', type=str,
                        choices=['rate', 'ttfs', 'phase', 'burst'], default='rate')
    parser.add_argument('--phase-weighted', action="store_true",
                        help="Weighted input spikes (used in phase neural coding)")
    parser.add_argument('--burst-n-max', type=int, default=5,
                        help="Maximum number of spikes for burst coding.")
    parser.add_argument('--burst-t-min', type=int, default=2,
                        help="Minimum time interval of spikes for burst coding.")
    parser.add_argument('--ttfs-normalize', action="store_true",
                        help="Normalize the TTFS neural coding")
    parser.add_argument('--ttfs-linear', action="store_true",
                        help="Linear latency code instead of log latency code (for ttfs neural coding).")
    parser.add_argument('--sacc-max-dx', type=int, default=20,
                        help="Max distance of translation for the saccade in the x axis")
    parser.add_argument('--sacc-max-dy', type=int, default=20,
                        help="Max distance of translation for the saccade in the y axis")
    parser.add_argument('--sacc-delta', type=float, default=0.1,
                        help="Value of the delta modulation.")
    parser.add_argument('--sacc-number', type=int, default=3,
                        help="Number of saccades performed in synchrony-based neural coding.")
    parser.add_argument('--sync-dx', type=int, default=2,
                        help="Distance (in pixel) of the movement for each saccade.")

    # random
    parser.add_argument('--timesteps', '-t', type=int, default=100,
                        help="Number of timesteps for one inference.")
    parser.add_argument('--height', type=int,
                        help="Height dimension of the input", default=176)
    parser.add_argument('--width', type=int,
                        help="Width dimension of the input", default=240)
    parser.add_argument('--on-off', action='store_true',
                        help="If used, performs an On-Off filtering pre-processing step before.")

    # training related
    parser.add_argument('--experiment', required=True,
                        type=str, help="Name of the experiment.")
    parser.add_argument('--debug', action='store_true',
                        help="If used, forces to overfit only one batch of the train split (to debug the network).")
    parser.add_argument('--mixed-precision', action="store_true",
                        help="Enables the Automatic Mixed Precision optimization from PyTorch.")

    return parser.parse_args()
