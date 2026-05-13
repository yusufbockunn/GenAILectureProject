"""
step3_train_pipeline.py — Step 3: Full Training & Validation Pipeline

This script trains the ViT model in TWO modes for comparison:
  1. "real_only"  — trained exclusively on NIH ChestX-ray14 images (Baseline)
  2. "hybrid"     — trained on NIH images + ComfyUI synthetic images

Outputs:
  • Best model checkpoints  → outputs/checkpoints/ (and models/baseline_vit.pth)
  • Training curves (per run) → outputs/plots/
  • Comparison plot           → outputs/plots/comparison_real_vs_hybrid.png
  • Training history (JSON)   → outputs/logs/

Run:
    python step3_train_pipeline.py
    python step3_train_pipeline.py --mode real_only     # Baseline only
    python step3_train_pipeline.py --mode hybrid        # Hybrid only
    python step3_train_pipeline.py --mode both          # Both (default)
    python step3_train_pipeline.py --epochs 10          # Override epochs
    python step3_train_pipeline.py --debug              # Fast-run mode (5 batches/epoch)
"""

import sys
import os
import argparse
import shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import random

from src.config import (
    DEVICE, NUM_CLASSES, MODEL_NAME, USE_AMP,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    EARLY_STOP_PATIENCE, SEED, PROJECT_ROOT, CHECKPOINT_DIR
)
from src.model import create_vit_model, get_optimizer_and_scheduler
from src.train import TrainingEngine, plot_comparison
from src.dataset import create_dataloaders

# Use only 30% of training data to speed up epochs while preserving class distribution
DATA_SUBSET_RATIO = 0.30


# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────

def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed set to {seed}\n")


# ──────────────────────────────────────────────
# SINGLE TRAINING RUN
# ──────────────────────────────────────────────

def run_training(
    run_name: str,
    include_synthetic: bool,
    epochs: int,
    freeze_backbone: bool = False,
    debug_mode: bool = False,
) -> dict:
    """Execute one complete training run."""
    print("\n" + "=" * 60)
    print(f"  🏁 Training Run: '{run_name}'")
    print(f"     Synthetic data: {'YES' if include_synthetic else 'NO'}")
    print("=" * 60 + "\n")

    # 1. Seed
    set_seed(SEED)

    # 2. Data (Subsetting training data to 30% via stratified sampling)
    train_loader, val_loader, test_loader, df = create_dataloaders(
        include_synthetic=include_synthetic,
        subset_ratio=DATA_SUBSET_RATIO,
    )

    # Compute Class Weights due to class imbalance
    labels = train_loader.dataset.labels
    counts = Counter(labels)
    total = len(labels)
    # class_weight = N / (C * n_i)
    weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"   ⚖️  Computed Class Weights: {class_weights.tolist()}\n")

    # Determine autocast device type (cuda or cpu)
    amp_device = DEVICE.type  # "cuda" or "cpu"
    amp_enabled = USE_AMP and amp_device == "cuda"

    # 3. Model
    model = create_vit_model(
        num_classes=NUM_CLASSES,
        model_name=MODEL_NAME,
        freeze_backbone=freeze_backbone,
    )

    # 4. Loss function with label smoothing AND class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 5. Optimizer & Scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    # 6. Training Engine
    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=amp_enabled,
        patience=EARLY_STOP_PATIENCE,
        run_name=run_name,
    )

    # 7. Run training (fast-run limits to 5 batches if debug_mode is True)
    max_batches = 5 if debug_mode else None
    result = engine.run(epochs=epochs, max_batches=max_batches)

    # Save a specific copy to models/baseline_vit.pth if this is the baseline run
    if run_name == "real_only":
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        best_model_path = CHECKPOINT_DIR / f"best_model_{run_name}.pt"
        dest_path = models_dir / "baseline_vit.pth"
        if best_model_path.exists():
            shutil.copy(best_model_path, dest_path)
            print(f"   💾 Baseline model explicitly saved to: {dest_path}")

    return result


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 3: ViT Training Pipeline")
    parser.add_argument(
        "--mode", type=str, default="real_only",  # Defaulting to real_only for baseline phase
        choices=["real_only", "hybrid", "both"],
        help="Training mode: 'real_only', 'hybrid', or 'both' (default: real_only)",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Max epochs per run (default: {EPOCHS})",
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze ViT backbone (train classifier head only)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Fast-Run Debug Mode (limits to 5 batches per epoch)",
    )
    args = parser.parse_args()

    print("\n" + "🚀 " * 20)
    print("  STEP 3 — Full Training & Validation Pipeline")
    print("🚀 " * 20)
    print(f"\n  Mode   : {args.mode}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Freeze : {args.freeze}")
    print(f"  Debug  : {args.debug}")
    print(f"  Device : {DEVICE}")
    print()

    results = {}

    # ── Run 1: Real data only (Baseline) ──
    if args.mode in ("real_only", "both"):
        results["real_only"] = run_training(
            run_name="real_only",
            include_synthetic=False,
            epochs=args.epochs,
            freeze_backbone=args.freeze,
            debug_mode=args.debug,
        )

    # ── Run 2: Real + Synthetic (hybrid) ──
    if args.mode in ("hybrid", "both"):
        results["hybrid"] = run_training(
            run_name="hybrid",
            include_synthetic=True,
            epochs=args.epochs,
            freeze_backbone=args.freeze,
            debug_mode=args.debug,
        )

    # ── Comparison plot (if both runs completed) ──
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("  📊 Generating comparison plot ...")
        print("=" * 60)
        plot_comparison(results)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  ✅  Step 3 Complete — Training Pipeline")
    print("=" * 60)

    for name, r in results.items():
        print(f"\n  Run: {name}")
        print(f"    Best epoch    : {r['best_epoch']}")
        print(f"    Best val loss : {r['best_val_loss']:.4f}")
        print(f"    Best val acc  : {r['best_val_acc']:.4f}")

    print(f"\n  📁 Checkpoints : outputs/checkpoints/ (and models/)")
    print(f"  📊 Plots       : outputs/plots/")
    print(f"  📄 Logs        : outputs/logs/")
    print(f"\n  ➡️  Confirm to proceed to Step 4")
    print("=" * 60)


if __name__ == "__main__":
    main()
