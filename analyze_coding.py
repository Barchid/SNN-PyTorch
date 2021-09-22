from typing import Tuple
from utils.oxford_iiit_pet_loader import OxfordPetDatasetLocalization, get_transforms, init_oxford_dataset

import snntorch.spikeplot as splt
import matplotlib.pyplot as plt


from torch.utils.data.dataloader import DataLoader
from utils.neural_coding import rate_coding, ttfs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    must_shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=2, shuffle=must_shuffle, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=2, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()

    data_batch, _, _ = next(iter(train_loader))
    data_batch = rate_coding(data_batch, timesteps=300)

    print(data_batch.shape)

    spikes = torch.squeeze(data_batch[:, 0, :, :])

    #  Plot animator
    fig, ax = plt.subplots()
    anim = splt.animator(spikes, fig, ax)
    anim.save("spike_pet_rate_anim.gif")
    plt.close()

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    #  s: size of scatter points; c: color of scatter points
    splt.raster(spikes, ax, s=1.5, c="black")
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()
