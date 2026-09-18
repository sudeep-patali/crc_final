"""
dataset.py
PyTorch Dataset + DataLoader utilities for both the binary (NOR/TUM) and
stage (stage0-3) classifiers. Handles:
  - Macenko stain normalization (applied once, cached optionally)
  - resizing + pixel normalization
  - H&E-safe augmentation via Albumentations
  - patient/slide-level stratified splitting to avoid data leakage
"""

import os
import re
import glob
import random
import numpy as np
import cv2
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from utils.stain_normalization import MacenkoNormalizer


# ----------------------------------------------------------------------
# Slide/patient ID extraction (for leakage-free splitting)
# ----------------------------------------------------------------------
def group_key_from_filename(fname):
    """
    Tries to recover a slide/patient identifier from a filename so that
    patches from the same slide never appear in both train and val/test.
    Falls back to the filename stem (no grouping) if no pattern matches.

    Adjust the regex below to match your actual TCGA/ STAC-9 naming scheme,
    e.g. TCGA-XX-XXXX-... or slideid_x123_y456.png
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"(TCGA-\w+-\w+)", stem)
    if m:
        return m.group(1)
    # common pattern: <slideid>_x<...>_y<...>
    m = re.match(r"(.+?)_x\d+_y\d+", stem)
    if m:
        return m.group(1)
    return stem


def slide_level_split(file_label_pairs, train=0.70, val=0.15, test=0.15, seed=42):
    """
    file_label_pairs: list of (filepath, label) tuples
    Returns: train_list, val_list, test_list  (each a list of (path,label))
    Splitting is stratified by class and grouped by slide/patient id.
    """
    random.seed(seed)
    by_class = defaultdict(list)
    for path, label in file_label_pairs:
        by_class[label].append(path)

    train_out, val_out, test_out = [], [], []

    for label, paths in by_class.items():
        groups = defaultdict(list)
        for p in paths:
            groups[group_key_from_filename(p)].append(p)

        group_keys = list(groups.keys())
        random.shuffle(group_keys)

        n = len(group_keys)
        n_train = max(1, int(n * train))
        n_val = max(1, int(n * val))

        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train:n_train + n_val]
        test_keys = group_keys[n_train + n_val:]

        for k in train_keys:
            train_out += [(p, label) for p in groups[k]]
        for k in val_keys:
            val_out += [(p, label) for p in groups[k]]
        for k in test_keys:
            test_out += [(p, label) for p in groups[k]]

    random.shuffle(train_out)
    random.shuffle(val_out)
    random.shuffle(test_out)
    return train_out, val_out, test_out


def collect_files(root_dir, class_names):
    """root_dir/class_name/*.png|jpg -> list of (path, class_index)"""
    pairs = []
    for idx, cname in enumerate(class_names):
        class_dir = os.path.join(root_dir, cname)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif"):
            for f in glob.glob(os.path.join(class_dir, ext)):
                pairs.append((f, idx))
    return pairs


# ----------------------------------------------------------------------
# Augmentations
# ----------------------------------------------------------------------
def get_train_transforms():
    return A.Compose([
        A.Resize(config.PATCH_SIZE, config.PATCH_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=1, sigma=15, p=0.2),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms():
    return A.Compose([
        A.Resize(config.PATCH_SIZE, config.PATCH_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class HistopathDataset(Dataset):
    def __init__(self, file_label_pairs, transform=None, use_stain_norm=True,
                 stain_normalizer=None):
        self.pairs = file_label_pairs
        self.transform = transform
        self.use_stain_norm = use_stain_norm
        self.stain_normalizer = stain_normalizer  # a fitted MacenkoNormalizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.use_stain_norm and self.stain_normalizer is not None:
            try:
                rgb = self.stain_normalizer.transform(rgb)
            except Exception:
                pass  # fall back to raw image if normalization fails on a tile

        if self.transform:
            augmented = self.transform(image=rgb)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


def build_dataloaders(root_dir, class_names, stain_normalizer=None,
                       batch_size=None, num_workers=None):
    """
    High-level helper: collects files, does slide-level split, builds
    train/val/test DataLoaders. Works identically for binary and stage data
    -- just pass the right root_dir + class_names.
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    pairs = collect_files(root_dir, class_names)
    if len(pairs) == 0:
        raise RuntimeError(f"No images found under {root_dir}. Check folder structure.")

    train_pairs, val_pairs, test_pairs = slide_level_split(
        pairs, config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT, config.SPLIT_SEED
    )

    train_ds = HistopathDataset(train_pairs, get_train_transforms(), True, stain_normalizer)
    val_ds   = HistopathDataset(val_pairs,   get_eval_transforms(),  True, stain_normalizer)
    test_ds  = HistopathDataset(test_pairs,  get_eval_transforms(),  True, stain_normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    print(f"[{os.path.basename(root_dir)}] train={len(train_ds)} "
          f"val={len(val_ds)} test={len(test_ds)}")

    return train_loader, val_loader, test_loader
