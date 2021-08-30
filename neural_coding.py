import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def rate_coding(image: np.ndarray):
    pass


def P_th(t: float, theta_0: float, tau_th: float):
    # calculate Pth(t)
    P_th = theta_0 * math.exp(-t/tau_th)
    return P_th


def ttfs(image: np.ndarray, theta_0: float = 1.0, tau_th: float = 6.0, timesteps: int = 300):
    # asserts
    assert len(image.shape) == 2  # force grayscale
    assert tau_th > 0.
    assert timesteps > 0

    # convert numpy array to float
    image = image.astype(np.float32)

    # Divide by max value of the image
    image = image / np.amax(image)

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
    ts = 100

    image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
    S = ttfs(image, theta_0, tau_th, ts)
    print(S.shape, np.unique(S))
    spikes = (S == 1.).sum()
    nonspikes = (S == 0.).sum()

    print(spikes, nonspikes)

    print(333 * 500 * 100 == spikes + nonspikes)

    print(333 * 500, spikes, abs(333 * 500 - spikes))
