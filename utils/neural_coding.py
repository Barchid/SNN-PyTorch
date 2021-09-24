import math

import snntorch
from utils.localization_utils import image_to_spikes
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.snn_utils import image2spiketrain
from snntorch import spikegen
import torch


def neural_coding(images: torch.Tensor, args) -> torch.Tensor:
    if args.neural_coding == 'rate':
        return rate_coding(images, args.timesteps)
    elif args.neural_coding == 'ttfs':
        return ttfs(images, timesteps=args.timesteps)
    elif args.neural_coding == 'phase':
        return phase_coding(images, timesteps=args.timesteps, is_weighted=args.phase_weighted)
    elif args.neural_coding == 'burst':
        return burst_coding(images, args.burst_n_max, args.timesteps, args.burst_t_min)


def burst_coding(images: torch.Tensor, N_max: int = 5, timesteps: int = 100, T_min: int = 2):
    # Compute N_s (the number of spikes per pixel)
    N_s = torch.ceil(N_max * images)

    # Compute ISI (the InterSpike Interval per pixel)
    ISI = torch.full_like(images, float(timesteps))

    ISI[N_s > 1.] = torch.ceil(-(timesteps - T_min)
                               * images[N_s > 1.] + timesteps)

    # Reconstruct the spikes tensor with N_s and ISI
    S = torch.zeros(
        (timesteps, images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
    # first timesteps are full of 0s until T_min

    distances = torch.zeros_like(ISI)  # separating two spikes for each pixels
    for i in range(timesteps):
        mask = torch.logical_and(distances == ISI, N_s > 0)
        S[i] = mask.float()
        distances += 1
        distances[mask] = 0
        N_s[mask] -= 1

    return S


def rate_coding(images: torch.Tensor, timesteps: int = 100):
    return spikegen.rate(images, num_steps=timesteps)
    # return image_to_spikes(images, max_duration=timesteps, input_shape=images.shape[1:])


def phase_coding(images: torch.Tensor, timesteps: int = 100, is_weighted: bool = False):
    """Function that converts a grayscale image into spiketrains following the phase neural coding
    presented in TODO

    Args:
        image (np.ndarray): the grayscale image to convert of dimension (B, C, H, W)
        timesteps (int, optional): the number of timesteps. Must be a multiple of 8 (required for the phases). Defaults to 100.
        is_weighted (bool, optional): Flag that indicates whether the output spikes are weighted following the w_s parameter defined in the original paper. Defaults to True.

    Returns:
        np.ndarray: the spike tensor of dimension (T, H, W)
    """
    # compute number of periods
    periods = (timesteps // 8) + 1

    # convert to numpy because we have to
    images = (images * 255).numpy().astype(np.uint8)

    # binary representation of the image (it makes 8 )
    bit_representation = np.unpackbits(
        images[..., None], axis=-1).transpose(-1, 0, 1, 2, 3).astype(np.float32)

    # IF the weighted input option is used
    if is_weighted:
        # obtained with the equation seen in the referenced paper
        w_s = [0.5, 0.25, 0.125, 0.0625, 0.0313,
               0.015625, 0.0078125, 0.00390625]
        for i, weight in enumerate(w_s):
            bit_representation[i, :, :, :,
                               :] = bit_representation[i, :, :, :, :] * weight

    # Repeat the bit representation to create the final output spikes
    S = np.tile(bit_representation, (periods, 1, 1, 1, 1))[
        0:timesteps, :, :, :, :]

    return torch.from_numpy(S)


def P_th(t: float, theta_0: float, tau_th: float):
    # calculate Pth(t)
    P_th = theta_0 * math.exp(-t/tau_th)
    return P_th


def ttfs_handmade(image: np.ndarray, theta_0: float = 1.0, tau_th: float = 6.0, timesteps: int = 300):
    """Implementation of Time-To-First Spike (TTFS) neural coding for a grayscale image, as defined in
    the paper : https://www.frontiersin.org/articles/10.3389/fnins.2021.638474/full#F2

    Args:
        image (np.ndarray): the grayscale image of dimension (H, W) to convert into spikes
        theta_0 (float, optional): the parameter theta_0 (see the paper). Defaults to 1.0.
        tau_th (float, optional): the time constant parameter to compute the threshold at each timestep (see the paper). Defaults to 6.0.
        timesteps (int, optional): the total number of timesteps for one image. Defaults to 300.

    Returns:
        np.ndarray: the neural encoded image of dimension (timesteps, H, W)
    """
    # asserts
    assert len(image.shape) == 2  # force grayscale
    assert tau_th > 0.
    assert timesteps > 0

    # Divide by max value of the image
    image = image.astype(np.float32) / np.amax(image)

    # output spikes
    S = np.zeros((timesteps, image.shape[0], image.shape[1]), dtype=np.float32)

    for t in range(timesteps):
        # calculate P_th for the current timestep t
        threshold = P_th(t, theta_0, tau_th)

        # generate the input spikes for the current timestep using the threshold
        mask = image >= threshold
        S[t][mask] = 1.0  # Spikes when the pixel value exceeds the threshold
        image[mask] = 0.

    return S


def ttfs(images: np.ndarray, timesteps: int):
    return spikegen.latency(images, num_steps=timesteps, normalize=True, linear=False)


if __name__ == '__main__':
    theta_0 = 1.0
    tau_th = 6.0
    ts = 80

    image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
    # S = ttfs(image, theta_0, tau_th, ts)
    S = phase_coding(image, ts, is_weighted=False)
    print(S.shape)
    exit()
    spikes = (S == 1.).sum()
    nonspikes = (S == 0.).sum()

    print(spikes, nonspikes)

    print('Spikes !!!!!')
    for t in range(ts):
        print((S[t] == 1.).sum())
