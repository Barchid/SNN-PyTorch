from os import times
from random import gauss
from typing import Tuple
from celluloid import Camera

from torch.utils import data
from utils.oxford_iiit_pet_loader import OxfordPetDatasetLocalization, get_transforms, init_oxford_dataset

import snntorch.spikeplot as splt
import matplotlib.pyplot as plt


from torch.utils.data.dataloader import DataLoader
from utils.neural_coding import burst_coding, phase_coding, rate_coding, saccade_coding, synchrony_coding, ttfs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    # Initialize Oxford-IIIT-Pet dataset in data/oxford_iiit_pet directory
    (
        train_images_filenames,
        val_images_filenames,
        images_directory,
        masks_directory,
        xml_directory,
    ) = init_oxford_dataset("data/oxford_augmented")
    # Transforms
    transform_train = get_transforms(88, 120, is_training=True)
    transform_val = get_transforms(80, 120, is_training=False)

    # Datasets
    train_data = OxfordPetDatasetLocalization(
        train_images_filenames,
        images_directory,
        masks_directory,
        transform=transform_train,
        use_DOG=True,  # TODO
        # gauss_sigma=0.5
    )

    val_data = OxfordPetDatasetLocalization(
        val_images_filenames,
        images_directory,
        masks_directory,
        transform=transform_val,
        use_DOG=False,
    )

    # Dataloaders
    must_shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=must_shuffle, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=2, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def anim_on_off(data, fig, ax, num_steps=False, interval=40):
    """Generate an animation from the data tensor of ON/OFF events (shape is (T, 2, H, W))

    Example::

        import snntorch.spikeplot as splt
        import matplotlib.pyplot as plt

        #  spike_data contains 128 samples, each of 100 time steps in duration
        print(spike_data.size())
        >>> torch.Size([100, 128, 2, 28, 28])

        #  Index into a single sample from a minibatch
        spike_data_sample = spike_data[:, 0, :, :, :]
        print(spike_data_sample.size())
        >>> torch.Size([100, 2, 28, 28])

        #  Plot
        fig, ax = plt.subplots()
        anim = splt.anim_on_off(spike_data_sample, fig, ax)
        HTML(anim.to_html5_video())

        #  Save as a gif
        anim.save("spike_ON_OFF.gif")

    :param data: Data tensor for a single sample across time steps of shape [num_steps x input_size]
    :type data: torch.Tensor

    :param fig: Top level container for all plot elements
    :type fig: matplotlib.figure.Figure

    :param ax: Contains additional figure elements and sets the coordinate system. E.g.:
        fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    :type ax: matplotlib.axes._subplots.AxesSubplot

    :param num_steps: Number of time steps to plot. If not specified, the number of entries in the first dimension
        of ``data`` will automatically be used, defaults to ``False``
    :type num_steps: int, optional

    :param interval: Delay between frames in milliseconds, defaults to ``40``
    :type interval: int, optional

    :param cmap: color map, defaults to ``plasma``
    :type cmap: string, optional

    :return: animation to be displayed using ``matplotlib.pyplot.show()``
    :rtype: FuncAnimation

    """

    if not num_steps:
        num_steps = data.size()[0]

    data = data.cpu()
    camera = Camera(fig)
    plt.axis("off")

    # iterate over time and take a snapshot with celluloid
    for step in range(num_steps):  # im appears unused but is required by camera.snap()
        # RGB frame where GREEN channel is used for ON events and RED channel is used for OFF events
        frame = torch.zeros((data.shape[2], data.shape[3], 3))
        frame[:, :, 0] += data[step, 0, :, :]
        frame[:, :, 1] += data[step, 1, :, :]
        _ = ax.imshow(frame)
        camera.snap()
    anim = camera.animate(interval=interval)

    return anim


if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()

    data_batch, _, _ = next(iter(train_loader))

    imgs = torch.squeeze(data_batch).numpy()
    print(data_batch.shape)

    synchro_batch = synchrony_coding(
        data_batch, timesteps=40, saccade_number=3)
    sacc_batch = saccade_coding(data_batch, timesteps=40, delta_threshold=0.2)
    rate_batch = rate_coding(data_batch, timesteps=40)
    ttfs_batch = ttfs(data_batch, timesteps=40, normalize=True, linear=False)
    phase_batch = phase_coding(data_batch, timesteps=40)
    burst_batch = burst_coding(data_batch, timesteps=40)

    # print(rate_batch.shape, ttfs_batch.shape,
    #       phase_batch.shape, burst_batch.shape)
    # print(sacc_batch.shape)
    print(synchro_batch.shape)

    # batch size iterations
    for i in range(1):  # range(ttfs_batch.shape[1]):
        print(f'reading batch {i}',imgs[i][0].shape)
        cv2.imwrite(f'mage_batch{i}.png',
                    (imgs[i][0] * 255).astype(np.uint8))

        spikes = torch.squeeze(synchro_batch[:, i, :, :])
        fig, ax = plt.subplots()
        anim = anim_on_off(spikes, fig, ax, interval=80)
        # anim = splt.animator(spikes, fig, ax, interval=80)
        anim.save(f"spike_synchro.mp4")

        spikes = torch.squeeze(sacc_batch[:, i, :, :])
        fig, ax = plt.subplots()
        anim = anim_on_off(spikes, fig, ax, interval=80)
        # anim = splt.animator(spikes, fig, ax, interval=80)
        anim.save(f"spike_saccades.mp4")

        #  Plot animator
        spikes = torch.squeeze(ttfs_batch[:, i, :, :])
        fig, ax = plt.subplots()
        anim = anim_on_off(spikes, fig, ax, interval=80)
        # anim = splt.animator(spikes, fig, ax, interval=80)
        anim.save(f"spike_ttfs.mp4")

        spikes = torch.squeeze(rate_batch[:, i, :, :])
        fig, ax = plt.subplots()
        anim = anim_on_off(spikes, fig, ax, interval=80)
        # anim = splt.animator(spikes, fig, ax, interval=80)
        anim.save(f"spike_rate.mp4")

        spikes = torch.squeeze(phase_batch[:, i, :, :])
        fig, ax = plt.subplots()
        anim = anim_on_off(spikes, fig, ax, interval=80)
        # anim = splt.animator(spikes, fig, ax, interval=80)
        anim.save(f"spike_phase.mp4")

        spikes = torch.squeeze(burst_batch[:, i, :, :])
        fig, ax = plt.subplots()
        anim = anim_on_off(spikes, fig, ax, interval=80)
        # anim = splt.animator(spikes, fig, ax, interval=80)
        anim.save(f"spike_burst.mp4")
