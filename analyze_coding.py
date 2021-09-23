from os import times
from typing import Tuple

from torch.utils import data
from utils.oxford_iiit_pet_loader import OxfordPetDatasetLocalization, get_transforms, init_oxford_dataset

import snntorch.spikeplot as splt
import matplotlib.pyplot as plt


from torch.utils.data.dataloader import DataLoader
from utils.neural_coding import phase_coding, rate_coding, ttfs
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
        use_DOG=False,  # TODO
    )

    val_data = OxfordPetDatasetLocalization(
        val_images_filenames,
        images_directory,
        masks_directory,
        transform=transform_val,
        use_DOG=False,
    )

    # Dataloaders
    must_shuffle = True
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=must_shuffle, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=2, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()

    data_batch, _, _ = next(iter(train_loader))

    imgs = torch.squeeze(data_batch).numpy()

    rate_batch = rate_coding(data_batch, timesteps=300)
    ttfs_batch = ttfs(data_batch, timesteps=300)
    phase_batch = phase_coding(data_batch, timesteps=300)

    print(rate_batch.shape, ttfs_batch.shape, phase_batch.shape)

    # batch_sum = torch.sum(data_batch, dim=0)

    # lol = torch.squeeze(batch_sum[0]).numpy()
    # mm = np.max(lol)
    # lol = lol/mm
    # lol = lol * 255
    # cv2.imwrite('debug.png', lol)

    # exit()

    # batch size iterations
    for i in range(ttfs_batch.shape[1]):
        print(f'reading batch {i}')
        cv2.imwrite(f'mage_batch{i}.png', (imgs[i] * 255).astype(np.uint8))
        spikes = torch.squeeze(phase_batch[:, i, :, :])
        #  Plot animator
        fig, ax = plt.subplots()
        anim = splt.animator(spikes, fig, ax)
        anim.save(f"spike_pet_rate_anim_batch{i}.mp4")
