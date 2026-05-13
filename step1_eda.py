"""
step1_eda.py — Step 1: Project Setup & Exploratory Data Analysis

This script:
  1. Loads and summarizes the NIH metadata CSV (if available)
  2. Scans real + synthetic image directories
  3. Visualizes class distribution (real vs synthetic)
  4. Shows sample images from each class
  5. Validates that the Dataset class works correctly

Run:
    python step1_eda.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for servers / headless)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from src.config import (
    CSV_PATH, REAL_DATA_DIR, SYNTH_DATA_DIR,
    TARGET_CLASSES, CLASS_TO_IDX, PLOT_DIR, SEED,
)
from src.dataset import (
    build_dataframe,
    create_dataloaders, get_eval_transforms,
    ChestXrayDataset,
)

# Reproducibility
np.random.seed(SEED)

# ──────────────────────────────────────────────
# 1. NIH METADATA CSV SUMMARY
# ──────────────────────────────────────────────

def summarize_csv():
    """Load the NIH metadata CSV and print basic statistics."""
    print("=" * 60)
    print("  📋  NIH ChestX-ray14 Metadata Summary")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"  ⚠  CSV not found at {CSV_PATH}")
        print("  ℹ  This is okay — we will load images from class subfolders directly.")
        print("  ℹ  Place the CSV here if you want metadata-level EDA.\n")
        return None

    df_csv = pd.read_csv(CSV_PATH)
    print(f"  Rows           : {len(df_csv):,}")
    print(f"  Columns        : {list(df_csv.columns)}")
    print(f"  Unique patients: {df_csv['Patient ID'].nunique():,}" if "Patient ID" in df_csv.columns else "")
    print()

    # Expand multi-label "Finding Labels" column
    if "Finding Labels" in df_csv.columns:
        all_labels = df_csv["Finding Labels"].str.split("|").explode()
        label_counts = all_labels.value_counts()
        print("  Label distribution (top 15):")
        for lbl, cnt in label_counts.head(15).items():
            marker = " ◀ TARGET" if lbl in TARGET_CLASSES else ""
            print(f"    {lbl:25s} : {cnt:>7,}{marker}")
        print()

    return df_csv


# ──────────────────────────────────────────────
# 2. FOLDER-BASED CLASS DISTRIBUTION
# ──────────────────────────────────────────────

def plot_class_distribution():
    """
    Build the unified DataFrame (real + synthetic) and plot
    side-by-side bar charts of class counts.
    """
    print("=" * 60)
    print("  📊  Class Distribution (Folder-Based)")
    print("=" * 60)

    # Real only
    df_real = build_dataframe(include_synthetic=False)

    # Real + Synthetic
    df_all = build_dataframe(include_synthetic=True)

    # ── Bar chart: Real vs Synthetic per class ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.color_palette("viridis", n_colors=len(TARGET_CLASSES))

    # Left: Real only
    real_counts = df_real["class_name"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    sns.barplot(x=real_counts.index, y=real_counts.values, ax=axes[0], palette=palette)
    axes[0].set_title("Real Images Only", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Class")
    for i, v in enumerate(real_counts.values):
        axes[0].text(i, v + max(real_counts.values) * 0.01, str(v),
                     ha="center", va="bottom", fontweight="bold")

    # Right: Real + Synthetic (stacked-style via hue)
    all_counts = df_all.groupby(["class_name", "source"]).size().reset_index(name="count")
    sns.barplot(
        data=all_counts, x="class_name", y="count", hue="source",
        ax=axes[1], palette={"real": "#2196F3", "synthetic": "#FF9800"},
        order=TARGET_CLASSES,
    )
    axes[1].set_title("Real + Synthetic Images", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Class")
    axes[1].legend(title="Source")

    plt.tight_layout()
    save_path = PLOT_DIR / "class_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✅ Saved class distribution plot → {save_path}")
    plt.close(fig)

    # Print numeric summary
    print("\n  Numeric Summary:")
    summary = df_all.groupby(["class_name", "source"]).size().unstack(fill_value=0)
    summary["TOTAL"] = summary.sum(axis=1)
    print(summary.to_string(index=True))
    print()

    return df_all


# ──────────────────────────────────────────────
# 3. SAMPLE IMAGE GRID
# ──────────────────────────────────────────────

def plot_sample_images(df: pd.DataFrame, samples_per_class: int = 4):
    """
    Show a grid of sample images for each target class.
    """
    print("=" * 60)
    print("  🖼️  Sample Images per Class")
    print("=" * 60)

    from PIL import Image

    n_classes = len(TARGET_CLASSES)
    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(3 * samples_per_class, 3 * n_classes),
    )

    for row, cls_name in enumerate(TARGET_CLASSES):
        cls_df = df[df["class_name"] == cls_name]

        if len(cls_df) == 0:
            for col in range(samples_per_class):
                ax = axes[row, col] if n_classes > 1 else axes[col]
                ax.text(0.5, 0.5, "No images", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(cls_name, fontsize=12, fontweight="bold")
            continue

        sampled = cls_df.sample(n=min(samples_per_class, len(cls_df)), random_state=SEED)

        for col in range(samples_per_class):
            ax = axes[row, col] if n_classes > 1 else axes[col]
            if col < len(sampled):
                img_path = sampled.iloc[col]["image_path"]
                source   = sampled.iloc[col]["source"]
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                ax.imshow(img)
                ax.set_title(f"{source}", fontsize=9, color="green" if source == "real" else "orange")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=12, fontweight="bold")

    plt.suptitle("Sample X-ray Images by Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = PLOT_DIR / "sample_images.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved sample images grid → {save_path}\n")
    plt.close(fig)


# ──────────────────────────────────────────────
# 4. DATASET SANITY CHECK
# ──────────────────────────────────────────────

def sanity_check_dataset():
    """
    Verify that the DataLoader pipeline works end-to-end:
    load one batch, print shapes, and confirm labels are valid.
    """
    print("=" * 60)
    print("  🧪  Dataset Sanity Check")
    print("=" * 60)

    try:
        train_loader, val_loader, test_loader, df = create_dataloaders(include_synthetic=True)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        print("  ℹ  Place your images in class subfolders under data/real/ and data/synthetic/")
        return

    # Grab one batch
    images, labels = next(iter(train_loader))
    print(f"  Batch image shape : {images.shape}")    # Expected: [B, 3, 224, 224]
    print(f"  Batch label shape : {labels.shape}")     # Expected: [B]
    print(f"  Label range       : {labels.min().item()} – {labels.max().item()}")
    print(f"  Pixel value range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"  DataLoader sizes  : train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)} batches")
    print("  ✅ Dataset sanity check passed!\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("  STEP 1 — Project Setup & Exploratory Data Analysis")
    print("🚀 " * 20 + "\n")

    # 1) CSV summary (optional — works even without CSV)
    summarize_csv()

    # 2) Class distribution plot
    df_all = plot_class_distribution()

    # 3) Sample image grid
    if df_all is not None and len(df_all) > 0:
        plot_sample_images(df_all)

    # 4) Sanity check
    if df_all is not None and len(df_all) > 0:
        sanity_check_dataset()

    print("=" * 60)
    print("  ✅  Step 1 Complete — Review plots in outputs/plots/")
    print("  ➡️  When ready, confirm to proceed to Step 2 (Model Setup)")
    print("=" * 60)
