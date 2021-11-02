from typing import List
import math
from celluloid import Camera
import cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch


class SAM(object):
    """Some Information about SAM"""

    def __init__(self, layer: nn.Module, input_height: int, input_width: int, gamma=0.4):
        super(SAM, self).__init__()
        self.hook = layer.register_forward_hook(self.hook_save_spikes)
        self.spike_rec = []
        self.gamma = gamma
        self.height = input_height
        self.width = input_width

    def hook_save_spikes(self, module, input, output):
        spikes = output[0].detach().cpu().numpy()
        self.spike_rec.append(spikes)

    def get_sam(self):
        # Compute the SAM for each layer and each timesteps
        heatmaps = []

        # FOR EACH timesteps
        for t in range(1, len(self.spike_rec)):
            NCS = np.zeros_like(self.spike_rec[0])

            # previous timesteps (i.e. t_p < t)
            for t_p in range(0, t):
                mask = self.spike_rec[t_p] == 1.
                NCS[mask] += math.exp(-self.gamma * abs(t - t_p))

            M = np.sum(NCS * self.spike_rec[t], axis=1)
            heatmap = self._format_heatmap(M)
            heatmaps.append(heatmap)

        # Resets the spike_rec
        self.spike_rec = []

        return heatmaps

    def _format_heatmap(self, M: np.ndarray):
        batch = []

        # for each heatmap in the batch
        for i in range(M.shape[0]):

            # normalize between 0 and 1
            max = np.max(M[i])
            min = np.min(M[i])
            heatmap = (M[i] - min) / (max - min + 1e-7)

            # resize the heatmap
            heatmap = cv2.resize(heatmap, (self.width, self.height))

            batch.append(heatmap)

        return np.array(batch)

    def release(self):
        self.hook.remove()


def heatmap_video(original_image: np.ndarray, heatmaps: List[np.ndarray], filename: str):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    plt.axis("off")

    for heatmap in heatmaps:
        img_heatmap = show_cam_on_image(original_image, heatmap, use_rgb=True)

        ax.imshow(img_heatmap)
        camera.snap()

    anim = camera.animate(interval=40)
    anim.save(filename)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
