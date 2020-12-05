# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from PIL import Image
import torch.utils.data as data
import os

class ListDataset(data.Dataset):
    def __init__(self, img_dir, image_lists, category_lists, transforms=None):
        self.img_dir = img_dir
        self.image_lists = image_lists
        self.category_lists = category_lists
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.image_lists[idx])).convert("RGB")
        target = self.category_lists[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass

class ListDatasetVal(data.Dataset):
    def __init__(self, img_dir, image_lists, category_lists, transforms=None):
        self.img_dir = img_dir
        self.image_lists = image_lists
        self.category_lists = category_lists
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_lists[idx])
        img = Image.open(img_path).convert("RGB")
        target = self.category_lists[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, self.image_lists[idx]

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass
