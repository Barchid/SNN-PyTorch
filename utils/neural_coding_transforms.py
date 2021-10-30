import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.neural_coding import phase_coding, rate_coding, saccade_coding, synchrony_coding, ttfs


class SaccadeCoding(object):
    def __init__(self, timesteps: int, max_dx: int = 20, max_dy: int = 20, delta_threshold=0.1):
        self.timesteps = timesteps
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.delta_threshold = delta_threshold

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return saccade_coding(images, self.timesteps, self.max_dx, self.max_dy, self.delta_threshold)


class SynchronyCoding(object):
    def __init__(self, timesteps: int, saccade_number: int = 3, dx: int = 2, delta_threshold: float = 0.1):
        self.timesteps = timesteps
        self.saccade_number = saccade_number
        self.dx = dx
        self.delta_threshold = delta_threshold

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return synchrony_coding(images, self.timesteps, self.saccade_number, self.delta_threshold, self.dx)


class RateCoding(object):
    def __init__(self, timesteps: int):
        self.timesteps = timesteps

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return rate_coding(images, self.timesteps)


class PhaseCoding(object):
    def __init__(self, timesteps: int, is_weighted: bool = False):
        self.timesteps = timesteps
        self.is_weighted = is_weighted

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return phase_coding(images, self.timesteps, self.is_weighted)


class TTFSCoding(object):
    def __init__(self, timesteps: int, normalize: bool, linear: bool):
        self.timesteps = timesteps
        self.normalize = normalize
        self.linear = linear

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return ttfs(images, self.timesteps, self.normalize, self.linear)


def get_neural_coding(args):
    if args.neural_coding == 'rate':
        return RateCoding(args.timesteps)
    elif args.neural_coding == 'ttfs':
        return TTFSCoding(args.timesteps, args.ttfs_normalize, args.ttfs_linear)
    elif args.neural_coding == 'phase':
        return PhaseCoding(args.timesteps, args.phase_weighted)
    # elif args.neural_coding == 'burst':
    #     return burst_coding(images, args.burst_n_max, args.timesteps, args.burst_t_min)
    elif args.neural_coding == 'saccades':
        return SaccadeCoding(args.timesteps, args.sacc_max_dx, args.sacc_max_dy, args.sacc_delta)
    elif args.neural_coding == 'synchrony':
        return SynchronyCoding(args.timesteps, args.sacc_number, args.sync_dx, args.sacc_delta)
    else:
        raise NotImplementedError()
