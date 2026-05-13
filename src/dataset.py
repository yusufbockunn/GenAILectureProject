"""
dataset.py — PyTorch Dataset & DataLoader factories for the Chest X-ray project.

Data layout:
  • Real NIH images are in a FLAT folder (data/images-224/images-224/*.png)
  • Labels come from Data_Entry_2017.csv ("Finding Labels" column, pipe-separated)
  • Official train/test split comes from train_val_list_NIH.txt / test_list_NIH.txt
  • Synthetic images (data/images-s/) can be added to augment minority classes

Supports two modes:
  • Real-only:         uses only NIH images
  • Real + Synthetic:  merges NIH images with ComfyUI-generated synthetic images
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.config import (
    REAL_IMAGE_DIR, SYNTH_IMAGE_DIR, CSV_PATH,
    TRAIN_VAL_LIST_PATH, TEST_LIST_PATH,
    TARGET_CLASSES, CLASS_TO_IDX, IMG_SIZE,
    BATCH_SIZE, NUM_WORKERS, SEED, VAL_RATIO,
)


# ──────────────────────────────────────────────
# 1. TRANSFORMS
# ──────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Augmentations + normalization for training split."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats (ViT pretrained on ImageNet)
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic transforms for validation / test."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ──────────────────────────────────────────────
# 2. DATASET CLASS
# ──────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    A flexible PyTorch Dataset for the Chest X-ray classification task.

    Parameters
    ----------
    image_paths : list of str/Path
        Absolute paths to each image file.
    labels : list of int
        Integer class labels corresponding to each image.
    transform : torchvision.transforms.Compose, optional
        Transforms to apply to each image.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image → convert to RGB (X-rays are often grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ──────────────────────────────────────────────
# 3. CSV-BASED DATA LOADING
# ──────────────────────────────────────────────

def load_nih_metadata() -> pd.DataFrame:
    """
    Load NIH CSV and filter to single-label images in our TARGET_CLASSES.

    The NIH dataset is multi-label (e.g. "Atelectasis|Effusion").
    For clean classification, we ONLY keep images with exactly ONE
    label that matches our target classes.

    Returns
    -------
    df : pd.DataFrame
        Columns: filename, class_name, label (int), image_path, source
    """
    print(f"📋 Loading NIH metadata from {CSV_PATH} ...")
    df_csv = pd.read_csv(CSV_PATH)
    print(f"   Total rows in CSV: {len(df_csv):,}")

    # Keep only single-label rows whose label is in TARGET_CLASSES
    # (Exclude multi-label rows like "Atelectasis|Effusion")
    mask_single = ~df_csv["Finding Labels"].str.contains(r"\|", regex=True)
    mask_target = df_csv["Finding Labels"].isin(TARGET_CLASSES)
    df_filtered = df_csv[mask_single & mask_target].copy()

    df_filtered = df_filtered.rename(columns={"Image Index": "filename", "Finding Labels": "class_name"})
    df_filtered["label"] = df_filtered["class_name"].map(CLASS_TO_IDX)
    df_filtered["image_path"] = df_filtered["filename"].apply(
        lambda fn: str(REAL_IMAGE_DIR / fn)
    )
    df_filtered["source"] = "real"

    # Keep only images that actually exist on disk
    df_filtered["exists"] = df_filtered["image_path"].apply(lambda p: os.path.isfile(p))
    n_missing = (~df_filtered["exists"]).sum()
    if n_missing > 0:
        print(f"   ⚠  {n_missing} images referenced in CSV but missing from disk — skipping them.")
    df_filtered = df_filtered[df_filtered["exists"]].drop(columns=["exists"])

    df_filtered = df_filtered[["filename", "class_name", "label", "image_path", "source"]].reset_index(drop=True)

    print(f"   After filtering to {TARGET_CLASSES}:")
    for cls in TARGET_CLASSES:
        cnt = (df_filtered["class_name"] == cls).sum()
        print(f"     {cls:20s} : {cnt:>6,}")
    print(f"   Total usable images: {len(df_filtered):,}\n")

    return df_filtered


def load_synthetic_images() -> pd.DataFrame:
    """
    Scan the synthetic image directory.

    Expected layout (either flat or class-subfolder):
      Flat:       data/images-s/synth_001.png  (assumes ALL are Consolidation)
      Subfolder:  data/images-s/Consolidation/synth_001.png

    Returns
    -------
    df : pd.DataFrame  (same schema as load_nih_metadata output)
    """
    if not SYNTH_IMAGE_DIR.exists():
        print(f"📂 Synthetic dir not found: {SYNTH_IMAGE_DIR} — skipping.\n")
        return pd.DataFrame(columns=["filename", "class_name", "label", "image_path", "source"])

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    records = []

    # Check if there are class subfolders
    subdirs = [d for d in SYNTH_IMAGE_DIR.iterdir() if d.is_dir() and d.name in TARGET_CLASSES]

    if subdirs:
        # Subfolder mode: data/images-s/<ClassName>/*.png
        print(f"📂 Scanning synthetic images (subfolder mode) in {SYNTH_IMAGE_DIR} ...")
        for cls_dir in subdirs:
            cls_name = cls_dir.name
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() in valid_ext:
                    records.append({
                        "filename": fpath.name,
                        "class_name": cls_name,
                        "label": CLASS_TO_IDX[cls_name],
                        "image_path": str(fpath),
                        "source": "synthetic",
                    })
    else:
        # Flat mode: assume ALL images are Consolidation (minority class augmentation)
        print(f"📂 Scanning synthetic images (flat mode, assuming Consolidation) in {SYNTH_IMAGE_DIR} ...")
        cls_name = "Consolidation"
        for fpath in SYNTH_IMAGE_DIR.iterdir():
            if fpath.suffix.lower() in valid_ext:
                records.append({
                    "filename": fpath.name,
                    "class_name": cls_name,
                    "label": CLASS_TO_IDX[cls_name],
                    "image_path": str(fpath),
                    "source": "synthetic",
                })

    df_synth = pd.DataFrame(records)
    print(f"   Found {len(df_synth)} synthetic images.\n")
    return df_synth


def load_split_lists() -> Tuple[set, set]:
    """
    Load the official NIH train/val and test filename lists.

    Returns
    -------
    train_val_filenames : set of str
    test_filenames      : set of str
    """
    train_val_fns = set()
    test_fns = set()

    if TRAIN_VAL_LIST_PATH.exists():
        with open(TRAIN_VAL_LIST_PATH, "r") as f:
            train_val_fns = {line.strip() for line in f if line.strip()}
        print(f"   Train/Val list: {len(train_val_fns):,} filenames")

    if TEST_LIST_PATH.exists():
        with open(TEST_LIST_PATH, "r") as f:
            test_fns = {line.strip() for line in f if line.strip()}
        print(f"   Test list     : {len(test_fns):,} filenames")

    return train_val_fns, test_fns


# ──────────────────────────────────────────────
# 4. BUILD DATAFRAME
# ──────────────────────────────────────────────

def build_dataframe(include_synthetic: bool = False) -> pd.DataFrame:
    """
    Build a unified DataFrame of all images (real + optionally synthetic).

    Columns: filename, class_name, label (int), image_path, source
    """
    df_real = load_nih_metadata()

    if not include_synthetic:
        return df_real

    df_synth = load_synthetic_images()
    df = pd.concat([df_real, df_synth], ignore_index=True)
    print(f"📊 Combined dataset: {len(df):,} images "
          f"({len(df_real):,} real + {len(df_synth):,} synthetic)\n")
    return df


# ──────────────────────────────────────────────
# 5. SPLIT & DATALOADER FACTORY
# ──────────────────────────────────────────────

def create_dataloaders(
    include_synthetic: bool = False,
    subset_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    End-to-end pipeline:
      1. Load NIH metadata + optionally synthetic images → DataFrame
      2. Split using official NIH train/test lists + stratified val split
      3. Synthetic images are added ONLY to the training set (not val/test)
      4. Wrap in Dataset objects with appropriate transforms
      5. Return DataLoaders + the full DataFrame

    Returns
    -------
    train_loader, val_loader, test_loader, df
    """
    # Load all real images
    df_real = load_nih_metadata()

    if len(df_real) == 0:
        raise RuntimeError(
            f"No usable images found! Check that {REAL_IMAGE_DIR} contains the NIH images "
            f"and {CSV_PATH} exists."
        )

    # --- Official train/val vs test split ---
    print("📂 Loading official NIH split lists ...")
    train_val_fns, test_fns = load_split_lists()

    if train_val_fns and test_fns:
        # Use official split
        df_trainval = df_real[df_real["filename"].isin(train_val_fns)].reset_index(drop=True)
        df_test     = df_real[df_real["filename"].isin(test_fns)].reset_index(drop=True)
        print(f"   Official split — TrainVal: {len(df_trainval):,} | Test: {len(df_test):,}")
    else:
        # Fallback: random 85/15 split
        print("   ⚠  Split files not found — using random 85/15 split.")
        from sklearn.model_selection import train_test_split as tts
        df_trainval, df_test = tts(
            df_real, test_size=0.15, random_state=SEED,
            stratify=df_real["label"].values,
        )

    # --- Further split train_val → train + val ---
    train_idx, val_idx = train_test_split(
        np.arange(len(df_trainval)),
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=df_trainval["label"].values,
    )

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    if subset_ratio < 1.0:
        print(f"   ✂️  Subsetting real training data to {subset_ratio * 100:.0f}% (Stratified)...")
        df_train, _ = train_test_split(
            df_train,
            train_size=subset_ratio,
            random_state=SEED,
            stratify=df_train["label"].values,
        )
        df_train = df_train.reset_index(drop=True)

    # --- Add synthetic images to training set ONLY ---
    if include_synthetic:
        df_synth = load_synthetic_images()
        if len(df_synth) > 0:
            df_train = pd.concat([df_train, df_synth], ignore_index=True)
            print(f"   ➕ Added {len(df_synth)} synthetic images to training set.")

    print(f"\n📊 Final split sizes:")
    print(f"   Train : {len(df_train):,}  (real{' + synthetic' if include_synthetic else ''})")
    print(f"   Val   : {len(df_val):,}  (real only)")
    print(f"   Test  : {len(df_test):,}  (real only)\n")

    # --- Build Datasets ---
    train_ds = ChestXrayDataset(
        image_paths=df_train["image_path"].tolist(),
        labels=df_train["label"].tolist(),
        transform=get_train_transforms(),
    )
    val_ds = ChestXrayDataset(
        image_paths=df_val["image_path"].tolist(),
        labels=df_val["label"].tolist(),
        transform=get_eval_transforms(),
    )
    test_ds = ChestXrayDataset(
        image_paths=df_test["image_path"].tolist(),
        labels=df_test["label"].tolist(),
        transform=get_eval_transforms(),
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Full DF for reference
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    return train_loader, val_loader, test_loader, df_all
