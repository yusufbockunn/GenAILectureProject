"""
explainability.py — Attention-based Explainability for Vision Transformer (ViT).

Implements two complementary XAI techniques:

  1. **Attention Rollout** — Recursively multiplies attention matrices across all
     ViT layers to produce a single aggregated attention map. This captures how
     information flows from image patches to the final [CLS] token.

  2. **Grad-CAM for ViT** — Uses gradient-weighted attention from the last layer
     to highlight regions most relevant to the predicted class.

Both methods produce a spatial heatmap that is overlaid on the original X-ray image.

References:
  • Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
  • Chefer et al., "Transformer Interpretability Beyond Attention Visualization", CVPR 2021
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.config import DEVICE, IMG_SIZE, IDX_TO_CLASS, PLOT_DIR

# ──────────────────────────────────────────────
# 1. ATTENTION EXTRACTION
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_attentions(
    model: nn.Module,
    pixel_values: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Forward pass that captures attention weights from every ViT layer.

    Parameters
    ----------
    model : ViTForImageClassification
    pixel_values : torch.Tensor  [1, 3, 224, 224]

    Returns
    -------
    logits : torch.Tensor  [1, num_classes]
    attentions : list of torch.Tensor
        Each element has shape [1, num_heads, seq_len, seq_len]
        where seq_len = num_patches + 1 (for the [CLS] token).
        For ViT-Base/16 with 224×224 input: seq_len = 197 (196 patches + 1 CLS).
    """
    model.eval()
    outputs = model(pixel_values=pixel_values, output_attentions=True)
    logits = outputs.logits           # [1, num_classes]
    attentions = outputs.attentions   # tuple of [1, num_heads, seq_len, seq_len]
    return logits, list(attentions)


# ──────────────────────────────────────────────
# 2. ATTENTION ROLLOUT
# ──────────────────────────────────────────────

def attention_rollout(
    attentions: List[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """
    Compute Attention Rollout across all ViT layers.

    The idea: attention in each layer is a probability distribution.
    By recursively multiplying the attention matrices (with residual
    connections modeled as identity matrices), we get the total
    "attention flow" from the [CLS] token to each image patch.

    Parameters
    ----------
    attentions : list of torch.Tensor
        Attention weights from each layer, shape [1, heads, seq, seq].
    head_fusion : str
        How to aggregate across attention heads: 'mean', 'max', or 'min'.
    discard_ratio : float
        Fraction of lowest-attention values to zero out per layer (noise removal).

    Returns
    -------
    mask : np.ndarray  [grid_h, grid_w]
        Spatial attention map (values in [0, 1]).
    """
    result = None

    for layer_att in attentions:
        # layer_att: [1, num_heads, seq_len, seq_len] → [num_heads, seq_len, seq_len]
        att = layer_att.squeeze(0).cpu()

        # Fuse heads
        if head_fusion == "mean":
            att_fused = att.mean(dim=0)
        elif head_fusion == "max":
            att_fused = att.max(dim=0).values
        elif head_fusion == "min":
            att_fused = att.min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Discard low-attention noise
        if discard_ratio > 0:
            flat = att_fused.flatten()
            threshold = torch.quantile(flat, discard_ratio)
            att_fused = torch.where(att_fused > threshold, att_fused, torch.zeros_like(att_fused))
            # Re-normalize rows
            att_fused = att_fused / (att_fused.sum(dim=-1, keepdim=True) + 1e-9)

        # Add identity (residual connection)
        I = torch.eye(att_fused.size(0))
        att_with_residual = (att_fused + I) / 2.0

        # Re-normalize rows to sum to 1
        att_with_residual = att_with_residual / att_with_residual.sum(dim=-1, keepdim=True)

        # Recursive multiplication
        if result is None:
            result = att_with_residual
        else:
            result = torch.matmul(att_with_residual, result)

    # Extract attention from [CLS] token (index 0) to all patch tokens (indices 1:)
    cls_attention = result[0, 1:]  # [num_patches]

    # Reshape to spatial grid
    num_patches = cls_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))
    mask = cls_attention.reshape(grid_size, grid_size).numpy()

    # Normalize to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)

    return mask


# ──────────────────────────────────────────────
# 3. GRAD-CAM FOR ViT
# ──────────────────────────────────────────────

def grad_cam_vit(
    model: nn.Module,
    pixel_values: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, int, torch.Tensor]:
    """
    Grad-CAM adapted for ViT: uses gradients of the target class w.r.t.
    the last layer's attention weights to produce a class-discriminative map.

    Parameters
    ----------
    model : ViTForImageClassification
    pixel_values : torch.Tensor [1, 3, 224, 224]
    target_class : int or None
        If None, uses the predicted class.

    Returns
    -------
    mask : np.ndarray [grid_h, grid_w]
    predicted_class : int
    logits : torch.Tensor
    """
    model.eval()
    # Enable gradient computation for this forward pass
    pixel_values = pixel_values.clone().requires_grad_(True)

    outputs = model(pixel_values=pixel_values, output_attentions=True)
    logits = outputs.logits
    attentions = outputs.attentions  # tuple of [1, heads, seq, seq]

    # Determine target class
    predicted_class = logits.argmax(dim=-1).item()
    if target_class is None:
        target_class = predicted_class

    # Backward pass for target class
    model.zero_grad()
    score = logits[0, target_class]
    score.backward(retain_graph=True)

    # Get last layer attention and its gradient
    last_att = attentions[-1]  # [1, heads, seq, seq]

    # Use gradient of the attention output w.r.t. the score
    # We take the attention from [CLS] (row 0) to all patches (cols 1:)
    att_cls = last_att[0, :, 0, 1:]  # [heads, num_patches]

    # Compute gradients of last attention layer
    grads = torch.autograd.grad(
        score, last_att, retain_graph=True, allow_unused=True
    )[0]  # [1, heads, seq, seq]

    if grads is None:
        # Fallback: just use attention weights without gradient weighting
        cam = att_cls.mean(dim=0).detach().cpu().numpy()
    else:
        grad_cls = grads[0, :, 0, 1:]  # [heads, num_patches]
        # Weight attention by gradients, average across heads
        cam = (att_cls * grad_cls).mean(dim=0).detach().cpu().numpy()

    # ReLU (keep only positive contributions)
    cam = np.maximum(cam, 0)

    # Reshape to grid
    grid_size = int(np.sqrt(cam.shape[0]))
    cam = cam.reshape(grid_size, grid_size)

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    return cam, predicted_class, logits.detach()


# ──────────────────────────────────────────────
# 4. HEATMAP OVERLAY
# ──────────────────────────────────────────────

def create_heatmap_overlay(
    original_image: np.ndarray,
    attention_mask: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Resize the attention mask to match the image and overlay it as a heatmap.

    Parameters
    ----------
    original_image : np.ndarray [H, W, 3]  (uint8, RGB)
    attention_mask : np.ndarray [grid_h, grid_w]  (float, 0–1)
    colormap : str
        Matplotlib colormap name.
    alpha : float
        Opacity of the heatmap overlay (0 = transparent, 1 = opaque).

    Returns
    -------
    overlay : np.ndarray [H, W, 3]  (uint8, RGB)
    """
    h, w = original_image.shape[:2]

    # Resize attention to image size (bilinear interpolation for smooth heatmap)
    mask_resized = cv2.resize(attention_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(mask_resized)[:, :, :3]  # [H, W, 3] float 0–1
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

    return overlay


# ──────────────────────────────────────────────
# 5. VISUALIZATION PIPELINE
# ──────────────────────────────────────────────

def load_and_preprocess_image(
    image_path: str,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load an image and prepare it for both visualization and model input.

    Returns
    -------
    original_rgb : np.ndarray [224, 224, 3] (uint8)
    pixel_values : torch.Tensor [1, 3, 224, 224] (normalized)
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    original_rgb = np.array(img_resized)

    # Model preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pixel_values = preprocess(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    return original_rgb, pixel_values


def visualize_attention(
    model: nn.Module,
    image_path: str,
    save_path: Optional[str] = None,
    methods: Tuple[str, ...] = ("rollout", "gradcam"),
    true_label: Optional[str] = None,
) -> dict:
    """
    Generate attention map visualizations for a single image.

    Parameters
    ----------
    model : ViTForImageClassification
    image_path : str
        Path to the input X-ray image.
    save_path : str or None
        If provided, save the figure to this path.
    methods : tuple
        Which methods to visualize: 'rollout', 'gradcam', or both.
    true_label : str or None
        Ground truth class name (for display only).

    Returns
    -------
    result : dict
        Keys: predicted_class, predicted_name, confidence, masks (dict of method→mask).
    """
    # Load and preprocess
    original_rgb, pixel_values = load_and_preprocess_image(image_path)

    # Get prediction + attentions
    logits, attentions = extract_attentions(model, pixel_values)
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1).item()
    pred_name = IDX_TO_CLASS[pred_idx]
    confidence = probs[0, pred_idx].item()

    masks = {}
    overlays = {}

    # Attention Rollout
    if "rollout" in methods:
        rollout_mask = attention_rollout(attentions, head_fusion="mean", discard_ratio=0.1)
        masks["rollout"] = rollout_mask
        overlays["rollout"] = create_heatmap_overlay(original_rgb, rollout_mask, colormap="jet", alpha=0.5)

    # Grad-CAM
    if "gradcam" in methods:
        gradcam_mask, _, _ = grad_cam_vit(model, pixel_values, target_class=pred_idx)
        masks["gradcam"] = gradcam_mask
        overlays["gradcam"] = create_heatmap_overlay(original_rgb, gradcam_mask, colormap="inferno", alpha=0.5)

    # ── Plot ──
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Original image
    axes[0].imshow(original_rgb)
    title_parts = [f"Pred: {pred_name} ({confidence:.1%})"]
    if true_label:
        title_parts.insert(0, f"True: {true_label}")
    axes[0].set_title("\n".join(title_parts), fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Attention maps
    for i, method in enumerate(methods):
        ax = axes[i + 1]
        ax.imshow(overlays[method])
        method_display = "Attention Rollout" if method == "rollout" else "Grad-CAM"
        ax.set_title(f"{method_display}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  💾 Saved → {save_path}")

    plt.close(fig)

    return {
        "predicted_class": pred_idx,
        "predicted_name": pred_name,
        "confidence": confidence,
        "masks": masks,
    }


def visualize_batch(
    model: nn.Module,
    image_paths: List[str],
    true_labels: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    methods: Tuple[str, ...] = ("rollout", "gradcam"),
) -> List[dict]:
    """
    Generate attention maps for a batch of images and save a summary grid.

    Parameters
    ----------
    model : ViTForImageClassification
    image_paths : list of str
    true_labels : list of str or None
    save_dir : str or None
    methods : tuple

    Returns
    -------
    results : list of dict
    """
    if save_dir is None:
        save_dir = str(PLOT_DIR / "attention_maps")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(image_paths):
        true_lbl = true_labels[i] if true_labels else None
        fname = Path(img_path).stem
        save_path = str(Path(save_dir) / f"attention_{fname}.png")

        result = visualize_attention(
            model=model,
            image_path=img_path,
            save_path=save_path,
            methods=methods,
            true_label=true_lbl,
        )
        results.append(result)

    # ── Summary grid ──
    n_images = len(image_paths)
    n_cols = len(methods) + 1  # original + each method
    fig, axes = plt.subplots(n_images, n_cols, figsize=(5 * n_cols, 4 * n_images))

    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, img_path in enumerate(image_paths):
        original_rgb, pixel_values = load_and_preprocess_image(img_path)
        logits, attentions = extract_attentions(model, pixel_values)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        pred_name = IDX_TO_CLASS[pred_idx]
        confidence = probs[0, pred_idx].item()

        # Original
        axes[row, 0].imshow(original_rgb)
        true_lbl = true_labels[row] if true_labels else "?"
        axes[row, 0].set_title(f"True: {true_lbl}\nPred: {pred_name} ({confidence:.1%})",
                               fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")

        # Each method
        for col, method in enumerate(methods):
            if method == "rollout":
                mask = attention_rollout(attentions, head_fusion="mean", discard_ratio=0.1)
                cmap = "jet"
            else:
                mask, _, _ = grad_cam_vit(model, pixel_values, target_class=pred_idx)
                cmap = "inferno"

            overlay = create_heatmap_overlay(original_rgb, mask, colormap=cmap, alpha=0.5)
            axes[row, col + 1].imshow(overlay)
            method_name = "Attention Rollout" if method == "rollout" else "Grad-CAM"
            axes[row, col + 1].set_title(method_name, fontsize=9, fontweight="bold")
            axes[row, col + 1].axis("off")

    plt.suptitle("ViT Attention Maps — Chest X-ray", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    grid_path = str(Path(save_dir) / "attention_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  📊 Summary grid saved → {grid_path}")

    return results
