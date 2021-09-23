from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve
from numpy.lib.utils import _makenamedict
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True
TRAIN_COUNT = 6000  # 10000
VAL_COUNT = 1300  # 2091

# get the class from the name of image files
class_dic = {
    "Abyssinian": 0,
    "american_bulldog": 1,
    "american_pit_bull_terrier": 1,
    "basset_hound": 1,
    "beagle": 1,
    "Bengal": 0,
    "Birman": 0,
    "Bombay": 0,
    "boxer": 1,
    "British_Shorthair": 0,
    "chihuahua": 1,
    "Egyptian_Mau": 0,
    "english_cocker_spaniel": 1,
    "english_setter": 1,
    "german_shorthaired": 1,
    "great_pyrenees": 1,
    "havanese": 1,
    "japanese_chin": 1,
    "keeshond": 1,
    "leonberger": 1,
    "Maine_Coon": 0,
    "miniature_pinscher": 1,
    "newfoundland": 1,
    "Persian": 0,
    "pomeranian": 1,
    "pug": 1,
    "Ragdoll": 0,
    "Russian_Blue": 0,
    "saint_bernard": 1,
    "samoyed": 1,
    "scottish_terrier": 1,
    "shiba_inu": 1,
    "Siamese": 0,
    "Sphynx": 0,
    "staffordshire_bull_terrier": 1,
    "wheaten_terrier": 1,
    "yorkshire_terrier": 1,
}


################################################################################################################
# Function to download & unzip Oxford-IIT-Pet dataset
################################################################################################################
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


def init_oxford_dataset(dataset_directory):
    ################################################################################################################
    # Download & extract dataset
    ################################################################################################################
    if os.path.isdir(os.path.join(dataset_directory, "images")) and os.path.isdir(
        os.path.join(dataset_directory, "images")
    ):
        print("OXFORD IIT dataset already exists. Skip download & extract")
    else:
        # images
        filepath = os.path.join(dataset_directory, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # segmentation masks
        filepath = os.path.join(dataset_directory, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

    ################################################################################################################
    # Split train and test sets
    ################################################################################################################
    root_directory = os.path.join(dataset_directory)
    images_directory = os.path.join(root_directory, "images")
    masks_directory = os.path.join(root_directory, "annotations", "trimaps")
    xml_directory = os.path.join(root_directory, "annotations", "xmls")

    images_filenames = list(sorted(os.listdir(images_directory)))
    correct_images_filenames = [
        i
        for i in images_filenames
        if cv2.imread(os.path.join(images_directory, i)) is not None
    ]

    random.seed(42)
    random.shuffle(correct_images_filenames)

    train_images_filenames = correct_images_filenames[:TRAIN_COUNT]
    val_images_filenames = correct_images_filenames[
        TRAIN_COUNT: (TRAIN_COUNT + VAL_COUNT)
    ]
    return (
        train_images_filenames,
        val_images_filenames,
        images_directory,
        masks_directory,
        xml_directory,
    )


################################################################################################################
# Utils functions
################################################################################################################
# Create a binary mask
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


def DOG_transform(image, sigma1=1.0, sigma2=3.0, kernel_size=5):
    g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                          sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                          sigmaX=sigma2, sigmaY=sigma2)

    DOG = g2 - g1
    # cv2.imshow('coucou', DOG)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # DOG[DOG>0]=255
    # DOG[DOG<=0]=0
    # DOG = DOG.astype(np.uint8)
    # print(DOG.dtype)
    return DOG


def get_transforms(height, width, is_training=False, is_grayscale=True):
    if is_training:
        return A.Compose(
            [
                A.Resize(height, width),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=(0.) if is_grayscale else (0.485, 0.456, 0.406),
                    std=(1.) if is_grayscale else (0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(
                    mean=(0.) if is_grayscale else (0.485, 0.456, 0.406),
                    std=(1.) if is_grayscale else (0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )

################################################################################################################
# Oxfor dataset
################################################################################################################


class OxfordPetDataset(Dataset):
    def __init__(
        self, images_filenames, images_directory, masks_directory, transform=None, use_DOG=False
    ):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.use_DOG = use_DOG

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.use_DOG:
            image = DOG_transform(image)

        # get class from classes directory
        class_id = get_class_from_filename(image_filename)

        mask = cv2.imread(
            os.path.join(self.masks_directory,
                         image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )
        mask = preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask, class_id


class OxfordPetDatasetCLASS(Dataset):
    def __init__(self, images_filenames, images_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        # process image
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get class from classes directory
        class_id = get_class_from_filename(image_filename)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, class_id


class OxfordPetDatasetLocalization(Dataset):
    def __init__(
        self, images_filenames, images_directory, masks_directory, transform=None, use_DOG=False, is_grayscale=True
    ):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.use_DOG = use_DOG
        self.is_grayscale = is_grayscale

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        if self.is_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, 2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_DOG:
            image = DOG_transform(image)

        # get class from classes directory
        class_id = get_class_from_filename(image_filename)

        mask = cv2.imread(
            os.path.join(self.masks_directory,
                         image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )

        mask = preprocess_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # get bounding box from mask
        x_min, y_min, x_max, y_max = get_bbox_from_mask(mask.numpy())

        return image, class_id, torch.Tensor([x_min, y_min, x_max, y_max])


def get_bbox_from_mask(mask):
    rows, cols = np.where(mask == 1.0)

    if rows.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # get coordinates of bbox
    y_min = np.float(np.min(rows))
    x_min = np.float(np.min(cols))
    y_max = np.float(np.max(rows))
    x_max = np.float(np.max(cols))

    # normalize in interval [0, 1]
    x_min = x_min / mask.shape[1]
    y_min = y_min / mask.shape[0]
    x_max = x_max / mask.shape[1]
    y_max = y_max / mask.shape[0]

    return x_min, y_min, x_max, y_max


def get_class_from_filename(filename):
    filename = filename.replace('_AUGMENTED', '')
    breed_name = filename[
        : filename.rfind("_")
    ]  # get the name of the breed by removing the useless part "_{number}.png"

    return class_dic[breed_name]  # use the class dictionary (0=cat and 1=dog)


def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose(
        [t for t in dataset.transform if not isinstance(
            t, (A.Normalize, ToTensorV2))]
    )
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    im = cv2.imread('dumas.PNG', cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (176, 240))
    DOG_transform(im)
