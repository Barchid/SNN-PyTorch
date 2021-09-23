import math
from utils.localization_utils import image_to_spikes
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.snn_utils import image2spiketrain


def burst_coding(image: np.ndarray, N_max: int = 5, T_max: int = 100, T_min: int = 2):
    # asserts
    assert len(image.shape) == 2  # force grayscale

    # normalization between [0, 1]
    image = image.astype(np.float32) / np.amax(image)

    # Compute N_s (the number of spikes per pixel)
    N_s = np.ceil(N_max * image)

    # Compute ISI (the InterSpike Interval per pixel)
    ISI = np.full_like(image, float(N_max))
    ISI[N_s > 1.] = np.ceil(-(T_max - T_min) * image + T_max)

    S = []
    # first timesteps are full of 0s until T_min
    for _ in range(T_min):
        S.append(np.zeros_like(image))


def rate_coding(image: np.ndarray, timesteps: int = 100):
    return image_to_spikes(image, max_duration=timesteps, input_shape=image.shape[1:])


def phase_coding(image: np.ndarray, timesteps: int = 100, is_weighted: bool = True):
    """Function that converts a grayscale image into spiketrains following the phase neural coding
    presented in TODO

    Args:
        image (np.ndarray): the grayscale image to convert of dimension (H, W)
        timesteps (int, optional): the number of timesteps. Must be a multiple of 8 (required for the phases). Defaults to 100.
        is_weighted (bool, optional): Flag that indicates whether the output spikes are weighted following the w_s parameter defined in the original paper. Defaults to True.

    Returns:
        np.ndarray: the spike tensor of dimension (T, H, W)
    """
    # asserts
    assert len(image.shape) == 2  # force grayscale
    # check the conversion is possible. 8 because this will be the number of phases for the bit representation of [0, 255]
    assert timesteps % 8 == 0

    # compute number of periods
    periods = timesteps // 8

    # binary representation of the image (it makes 8 )
    bit_representation = np.unpackbits(
        image[..., None], axis=-1).transpose(2, 0, 1).astype(np.float32)

    # IF the weighted input option is used
    if is_weighted:
        # obtained with the equation seen in the referenced paper
        w_s = [0.5, 0.25, 0.125, 0.0625, 0.0313,
               0.015625, 0.0078125, 0.00390625]
        for i, weight in enumerate(w_s):
            bit_representation[i, :, :] = bit_representation[i, :, :] * weight

    # Repeat the bit representation to create the final output spikes
    S = np.tile(bit_representation, (periods, 1, 1))

    return S


def P_th(t: float, theta_0: float, tau_th: float):
    # calculate Pth(t)
    P_th = theta_0 * math.exp(-t/tau_th)
    return P_th


def ttfs(image: np.ndarray, theta_0: float = 1.0, tau_th: float = 6.0, timesteps: int = 300):
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


if __name__ == '__main__':
    theta_0 = 1.0
    tau_th = 6.0
    ts = 80

    image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
    S = ttfs(image, theta_0, tau_th, ts)
    print(S.shape, np.unique(S))
    exit()
    S = phase_coding(image, ts, is_weighted=True)
    print(S.shape, )
    spikes = (S == 1.).sum()
    nonspikes = (S == 0.).sum()

    print(spikes, nonspikes)

    print('Spikes !!!!!')
    for t in range(ts):
        print((S[t] == 1.).sum())
