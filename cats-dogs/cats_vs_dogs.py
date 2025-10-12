###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc.
# (now owned by Analog Devices Inc.)
# All Rights Reserved.
#
###################################################################################################
"""
Cats and Dogs Dataset (Improved Version)
"""
import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as album
import cv2
import ai8x

import torch._dynamo  # add this

torch._dynamo.config.suppress_errors = True  #


class CatsvsDogs(Dataset):
    """
    Cats vs Dogs dataset
    https://www.kaggle.com/datasets/salader/dogs-vs-cats

    Args:
        root_dir (string): Root directory of dataset where folders exist.
        d_type (string): "train" or "test".
        transform (callable, optional): Torch transform to apply.
        resize_size (tuple): Image resize target (width, height).
        augment_data (bool): Whether to apply augmentation for training.
    """

    labels = ['cat', 'dog']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'cat': 'cats', 'dog': 'dogs'}

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(128, 128), augment_data=False):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'cats_vs_dogs', d_type)

        if not self.__check_catsvsdogs_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")

        self.__get_image_paths()

        # Albumentations transformations
        if d_type == 'train' and augment_data:
            self.album_transform = album.Compose([
                album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2 * min(resize_size))),
                album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            ])
        else:
            self.album_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2 * min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            ])

        self.transform = transform

    def __check_catsvsdogs_data_exist(self):
        """Check if dataset exists in expected folder structure."""
        return os.path.isdir(self.data_dir)

    def __print_download_manual(self):
        """Prints instructions for downloading dataset manually."""
        print("******************************************")
        print("Dataset not found. Please follow these steps:")
        print("1. Download dataset from Kaggle: "
              "https://www.kaggle.com/datasets/salader/dogs-vs-cats")
        print("2. Extract and place into 'data/cats_vs_dogs/' folder like so:")
        print("   data/cats_vs_dogs/train/cats/")
        print("   data/cats_vs_dogs/train/dogs/")
        print("   data/cats_vs_dogs/test/cats/")
        print("   data/cats_vs_dogs/test/dogs/")
        print("******************************************")

    def __get_image_paths(self):
        """Gather image paths with labels."""
        self.data_list = []
        for label in self.labels:
            folder_name = self.label_to_folder_map[label]
            image_dir = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(image_dir):
                print(f"[Warning] Missing folder: {image_dir}")
                continue
            for file_name in sorted(os.listdir(image_dir)):
                file_path = os.path.join(image_dir, file_name)
                if os.path.isfile(file_path):
                    self.data_list.append((file_path, self.label_to_id_map[label]))

        if len(self.data_list) == 0:
            print(f"[Error] No images found in: {self.data_dir}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """Fetch image and label by index."""
        try:
            label = torch.tensor(self.data_list[index][1], dtype=torch.int64)
            image_path = self.data_list[index][0]

            image = cv2.imread(image_path)
            if image is None:
                print(f"[Warning] Skipping unreadable image: {image_path}")
                return self.__getitem__((index + 1) % len(self.data_list))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                if self.album_transform:
                    image = self.album_transform(image=image)["image"]
            except Exception as e:
                print(f"[Warning] Albumentations failed on {image_path}: {e}")
                return self.__getitem__((index + 1) % len(self.data_list))

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"[Critical] Error reading sample at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self.data_list))


def get_catsvsdogs_dataset(data, load_train, load_test):
    """
    Load the Cats vs Dogs dataset.
    Returns train/test datasets resized to 128x128.
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])

    train_dataset = None
    test_dataset = None

    if load_train:
        train_dataset = CatsvsDogs(root_dir=data_dir, d_type='train',
                                   transform=transform, augment_data=True)
    if load_test:
        test_dataset = CatsvsDogs(root_dir=data_dir, d_type='test',
                                  transform=transform, augment_data=False)

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'cats_vs_dogs',
        'input': (3, 128, 128),
        'output': ('cat', 'dog'),
        'loader': get_catsvsdogs_dataset,
    },
]
