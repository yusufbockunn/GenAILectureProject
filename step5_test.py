"""
step5_test.py — Official NIH Test-Set Evaluation

Sequentially evaluates two trained checkpoints on the official NIH test split:
  • outputs/checkpoints/best_model_real_only.pt
  • outputs/checkpoints/best_model_hybrid.pt

For each model this script:
  1. Loads the checkpoint (handles both raw state_dict and nested 'model_state_dict' key)
  2. Runs inference over the full NIH test set
  3. Prints a sklearn classification_report
  4. Saves a seaborn confusion-matrix heatmap to outputs/plots/test_results/
  5. Collects the Consolidation-class Recall for the final comparison table

Usage:
    python step5_test.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
from src.config import (
    DEVICE, TARGET_CLASSES, NUM_CLASSES, CHECKPOINT_DIR, PLOT_DIR,
)
from src.model import create_vit_model
from src.dataset import create_dataloaders

warnings.filterwarnings("ignore")

# ── output directory ─────────────────────────────────────────────────────────
TEST_PLOT_DIR = PLOT_DIR / "test_results"
TEST_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── models to evaluate ───────────────────────────────────────────────────────
CHECKPOINTS = {
    "Real-Only":   CHECKPOINT_DIR / "best_model_real_only.pt",
    "Hybrid":      CHECKPOINT_DIR / "best_model_hybrid.pt",
}

# ── Consolidation index (for the summary table) ───────────────────────────────
CONSOLIDATION_IDX = TARGET_CLASSES.index("Consolidation")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """
    Load a checkpoint into *model*.

    Handles two common save formats:
      • Raw state dict:              torch.save(model.state_dict(), path)
      • Dict with nested key:       torch.save({'model_state_dict': ..., ...}, path)
    """
    print(f"\n📂 Loading checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Please run the training scripts first."
        )

    # Load to CPU first to avoid device-mapping issues, then move to DEVICE
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        # Try common nested keys in order of preference
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"   Using nested key '{key}'")
                break
        else:
            # Assume the whole dict IS the state dict
            state_dict = checkpoint
            print("   Treating checkpoint dict as a raw state dict")
    else:
        state_dict = checkpoint  # already a raw OrderedDict

    model.load_state_dict(state_dict, strict=True)
    print("   ✅ Checkpoint loaded successfully.")
    return model


def run_inference(model: torch.nn.Module, test_loader) -> tuple[list, list]:
    """Run model over the test loader; return (all_labels, all_preds)."""
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="   Inference", leave=False, unit="batch"):
            images = images.to(DEVICE)

            outputs = model(pixel_values=images)
            logits  = outputs.logits          # (B, num_classes)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds


def save_confusion_matrix(
    labels: list,
    preds:  list,
    model_name: str,
    save_dir: Path,
) -> None:
    """Generate and save a seaborn confusion-matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=16, fontweight="bold")

    # ── Raw counts ──
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
        ax=axes[0],
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Raw Counts", fontsize=13)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # ── Row-normalised (recall per class on diagonal) ──
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Row-Normalised (Recall per Class)", fontsize=13)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    save_path = save_dir / f"confusion_matrix_{safe_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Confusion matrix saved → {save_path}")


def extract_consolidation_recall(report_dict: dict) -> float:
    """Pull the Recall for the Consolidation class from a classification_report dict."""
    return report_dict.get("Consolidation", {}).get("recall", float("nan"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  STEP 5 — Official NIH Test-Set Evaluation")
    print("=" * 70)

    # ── 1. Load the test DataLoader ──────────────────────────────────────────
    print("\n[1/3] Building test DataLoader (real-only, no synthetic) ...")
    # We use real-only (include_synthetic=False) so the test set is always
    # the clean official NIH split, regardless of which model we test.
    _, _, test_loader, _ = create_dataloaders(include_synthetic=False)
    print(f"   Test batches: {len(test_loader)} | "
          f"Approx samples: {len(test_loader.dataset):,}")

    # ── 2. Evaluate each checkpoint ─────────────────────────────────────────
    print("\n[2/3] Evaluating checkpoints sequentially ...")

    summary_rows = []   # list of (model_name, consolidation_recall)

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 60}")

        # a) Instantiate a fresh model skeleton
        model = create_vit_model(num_classes=NUM_CLASSES)

        # b) Load the checkpoint weights
        model = load_checkpoint(model, ckpt_path)
        model = model.to(DEVICE)

        # c) Run inference
        print("   Running inference on test set ...")
        all_labels, all_preds = run_inference(model, test_loader)

        # d) Classification report
        report_str = classification_report(
            all_labels,
            all_preds,
            target_names=TARGET_CLASSES,
            digits=4,
            zero_division=0,
        )
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=TARGET_CLASSES,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        print(f"\n📋 Classification Report — {model_name}:\n")
        print(report_str)

        # e) Save confusion matrix heatmap
        save_confusion_matrix(all_labels, all_preds, model_name, TEST_PLOT_DIR)

        # f) Collect Consolidation recall
        consol_recall = extract_consolidation_recall(report_dict)
        summary_rows.append((model_name, consol_recall))

        # Explicitly free GPU/DirectML memory before loading the next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 3. Summary comparison table ─────────────────────────────────────────
    print("\n[3/3] Summary — Consolidation Recall (Primary Success Metric)")
    print("=" * 70)
    print(f"  {'Model':<25} {'Consolidation Recall':>22}")
    print(f"  {'-'*25} {'-'*22}")
    for model_name, recall in summary_rows:
        bar_len = int(recall * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {model_name:<25} {recall:>20.4f}   [{bar}]")
    print("=" * 70)

    # Determine which model is better
    if len(summary_rows) == 2 and not any(np.isnan(r) for _, r in summary_rows):
        best = max(summary_rows, key=lambda x: x[1])
        worse = min(summary_rows, key=lambda x: x[1])
        delta = best[1] - worse[1]
        print(
            f"\n  🏆  Best model for Consolidation Recall: [{best[0]}]"
            f" (+{delta:.4f} vs {worse[0]})"
        )

    print(f"\n  Confusion matrix plots saved to: {TEST_PLOT_DIR}")
    print("\n✅  Evaluation complete.\n")


if __name__ == "__main__":
    main()
