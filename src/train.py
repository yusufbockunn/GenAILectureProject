"""
train.py — Training engine with mixed-precision (AMP), early stopping,
checkpoint saving, and training curve logging.

Provides:
  • train_one_epoch()  — single epoch training pass
  • validate()         — single epoch validation pass
  • EarlyStopping      — patience-based early stopping monitor
  • TrainingEngine     — full orchestrator with history, checkpoints, and plots
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.config import DEVICE, USE_AMP, CHECKPOINT_DIR, PLOT_DIR, LOG_DIR


# ──────────────────────────────────────────────
# 1. EARLY STOPPING
# ──────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss hasn't improved for `patience` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum decrease in val loss to count as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss == float('inf'):
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def status(self) -> str:
        best_loss_str = f"{self.best_loss:.4f}" if self.best_loss != float('inf') else "None"
        return f"EarlyStopping(counter={self.counter}/{self.patience}, best={best_loss_str})"


# ──────────────────────────────────────────────
# 2. SINGLE-EPOCH FUNCTIONS
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    use_amp: bool = USE_AMP,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Returns
    -------
    metrics : dict
        {"loss": float, "accuracy": float}
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False, ncols=100)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        amp_device = "cuda" if DEVICE.type == "cuda" else "cpu"
        with autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

        if max_batches is not None and (i + 1) >= max_batches:
            break

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    use_amp: bool = USE_AMP,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run one full validation epoch (no gradient computation).

    Returns
    -------
    metrics : dict
        {"loss": float, "accuracy": float}
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Val   Epoch {epoch}", leave=False, ncols=100)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        amp_device = "cuda" if DEVICE.type == "cuda" else "cpu"
        with autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

        if max_batches is not None and (i + 1) >= max_batches:
            break

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {"loss": epoch_loss, "accuracy": epoch_acc}


# ──────────────────────────────────────────────
# 3. TRAINING ENGINE
# ──────────────────────────────────────────────

class TrainingEngine:
    """
    Full training orchestrator with:
      • AMP mixed-precision
      • Early stopping
      • Best-model checkpoint saving
      • Training curve plotting
      • JSON history logging

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    criterion : nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : LR scheduler (optional)
    use_amp : bool
    patience : int — early stopping patience
    run_name : str — identifier for this run (used in file names)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        use_amp: bool = USE_AMP,
        patience: int = 5,
        run_name: str = "default",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.run_name = run_name

        # AMP scaler
        self.scaler = GradScaler(enabled=use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience)

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # History for logging / plotting
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
            "lr": [],         "epoch_time": [],
        }

    def train_epoch(self, epoch: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run one training epoch and record metrics."""
        start = time.time()
        metrics = train_one_epoch(
            self.model, self.train_loader, self.criterion,
            self.optimizer, self.scheduler, self.scaler,
            epoch, self.use_amp, max_batches
        )
        elapsed = time.time() - start

        self.history["train_loss"].append(metrics["loss"])
        self.history["train_acc"].append(metrics["accuracy"])
        self.history["epoch_time"].append(elapsed)

        # Current LR (from first param group)
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["lr"].append(current_lr)

        return metrics

    def validate_epoch(self, epoch: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run one validation epoch and record metrics."""
        metrics = validate(
            self.model, self.val_loader, self.criterion,
            epoch, self.use_amp, max_batches
        )
        self.history["val_loss"].append(metrics["loss"])
        self.history["val_acc"].append(metrics["accuracy"])
        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint if this is the best validation performance."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch

            ckpt_path = CHECKPOINT_DIR / f"best_model_{self.run_name}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "run_name": self.run_name,
            }, ckpt_path)
            return True
        return False

    def print_epoch_summary(self, epoch: int, train_m: dict, val_m: dict, saved: bool = False):
        """Pretty-print one epoch's results."""
        lr = self.history["lr"][-1]
        t = self.history["epoch_time"][-1]
        save_marker = " ★ SAVED" if saved else ""
        es_status = self.early_stopping.status()
        print(
            f"  Epoch {epoch:3d} │ "
            f"Train Loss: {train_m['loss']:.4f}  Acc: {train_m['accuracy']:.4f} │ "
            f"Val Loss: {val_m['loss']:.4f}  Acc: {val_m['accuracy']:.4f} │ "
            f"LR: {lr:.2e} │ {t:.1f}s{save_marker}"
        )
        print(f"             │ {es_status}")

    def run(self, epochs: int, max_batches: Optional[int] = None) -> Dict:
        """
        Execute the full training loop for `epochs` epochs.

        Includes early stopping and checkpoint saving.

        Returns
        -------
        result : dict
            Summary of the run including best metrics and history.
        """
        print(f"\n{'='*60}")
        print(f"  🚀 Starting training run: '{self.run_name}'")
        print(f"     Epochs: {epochs} | AMP: {self.use_amp} | Device: {DEVICE}")
        print(f"     Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")
        if max_batches is not None:
            print(f"     DEBUG MODE ON: Limited to {max_batches} batches per epoch.")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch, max_batches)
            val_metrics = self.validate_epoch(epoch, max_batches)

            # Save best model
            saved = self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["accuracy"])

            self.print_epoch_summary(epoch, train_metrics, val_metrics, saved)

            # Early stopping check
            if self.early_stopping(val_metrics["loss"]):
                print(f"\n  ⏹️  Early stopping triggered at epoch {epoch}.")
                break

        print(f"\n  ✅ Training complete for '{self.run_name}'")
        print(f"     Best epoch: {self.best_epoch} | "
              f"Val Loss: {self.best_val_loss:.4f} | Val Acc: {self.best_val_acc:.4f}")
        print(f"     Checkpoint: {CHECKPOINT_DIR / f'best_model_{self.run_name}.pt'}\n")

        # Save history to JSON
        self.save_history()

        # Plot training curves
        self.plot_training_curves()

        return {
            "run_name": self.run_name,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

    def save_history(self):
        """Save training history to a JSON file."""
        log_path = LOG_DIR / f"history_{self.run_name}.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  📄 History saved → {log_path}")

    def plot_training_curves(self):
        """Generate and save loss & accuracy training curves."""
        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(epochs_range, self.history["train_loss"], "b-o", label="Train Loss", markersize=4)
        axes[0].plot(epochs_range, self.history["val_loss"], "r-o", label="Val Loss", markersize=4)
        axes[0].axvline(x=self.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (e{self.best_epoch})")
        axes[0].set_title(f"Loss — {self.run_name}", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs_range, self.history["train_acc"], "b-o", label="Train Acc", markersize=4)
        axes[1].plot(epochs_range, self.history["val_acc"], "r-o", label="Val Acc", markersize=4)
        axes[1].axvline(x=self.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (e{self.best_epoch})")
        axes[1].set_title(f"Accuracy — {self.run_name}", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(epochs_range, self.history["lr"], "g-o", markersize=4)
        axes[2].set_title(f"Learning Rate — {self.run_name}", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = PLOT_DIR / f"training_curves_{self.run_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Training curves saved → {save_path}")

    def load_best_model(self):
        """Load the best checkpoint back into self.model."""
        ckpt_path = CHECKPOINT_DIR / f"best_model_{self.run_name}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"  ✅ Loaded best model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
        else:
            print(f"  ⚠  Checkpoint not found: {ckpt_path}")


# ──────────────────────────────────────────────
# 4. COMPARISON PLOT HELPER
# ──────────────────────────────────────────────

def plot_comparison(results: Dict[str, Dict]):
    """
    Compare training curves from multiple runs side by side.

    Parameters
    ----------
    results : dict
        Keys are run names, values are dicts with 'history' key
        (as returned by TrainingEngine.run()).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"real_only": "#2196F3", "hybrid": "#FF9800"}

    for run_name, result in results.items():
        h = result["history"]
        epochs_range = range(1, len(h["train_loss"]) + 1)
        color = colors.get(run_name, None)

        axes[0].plot(epochs_range, h["val_loss"], "-o", label=f"{run_name}", color=color, markersize=4)
        axes[1].plot(epochs_range, h["val_acc"], "-o", label=f"{run_name}", color=color, markersize=4)

    axes[0].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = PLOT_DIR / "comparison_real_vs_hybrid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📊 Comparison plot saved → {save_path}")
