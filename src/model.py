"""
model.py — Vision Transformer (ViT) wrapper for Chest X-ray classification.

Uses Hugging Face `transformers` to load a pretrained ViT and adapts
the classification head for our target classes.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

from src.config import MODEL_NAME, NUM_CLASSES, DEVICE


def create_vit_model(
    num_classes: int = NUM_CLASSES,
    model_name: str = MODEL_NAME,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    Load a pretrained ViT and replace the classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    model_name : str
        Hugging Face model identifier (e.g. 'google/vit-base-patch16-224-in21k').
    freeze_backbone : bool
        If True, freeze all layers except the classification head.
        Useful for quick experiments or very small datasets.

    Returns
    -------
    model : nn.Module
        The ViT model ready for fine-tuning, moved to DEVICE.
    """
    print(f"🔧 Loading pretrained ViT: {model_name}")
    print(f"   Num classes: {num_classes} | Freeze backbone: {freeze_backbone}")

    import time
    
    # Load pretrained model with a NEW classification head
    # Added retry mechanism for transient network issues with Hugging Face Hub
    max_retries = 3
    model = None
    for attempt in range(max_retries):
        try:
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,   # head size will differ from pretrained
                **kwargs
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️ Download error: {e}. Retrying in 5 seconds... ({attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                print(f"   ❌ Failed to download model '{model_name}' after {max_retries} attempts.")
                raise e

    # Optionally freeze the backbone (everything except the classifier)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("   ❄️  Backbone frozen — only classifier head is trainable.")

    # Move to device
    model = model.to(DEVICE)

    # Print parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters — Total: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"   Device: {DEVICE}\n")

    return model


def get_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
):
    """
    Create AdamW optimizer with a linear-warmup + cosine-decay LR schedule.

    Parameters
    ----------
    model : nn.Module
    learning_rate : float
    weight_decay : float
    epochs : int
    steps_per_epoch : int
        Number of training batches per epoch (len(train_loader)).

    Returns
    -------
    optimizer, scheduler
    """
    # Use different LR for backbone vs classifier head
    classifier_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": learning_rate},
        {"params": classifier_params, "lr": learning_rate * 10},  # 10× LR for head
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Cosine annealing with warmup
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"⚙️  Optimizer: AdamW | Backbone LR: {learning_rate} | Head LR: {learning_rate * 10}")
    print(f"   Scheduler: Linear warmup ({warmup_steps} steps) + Cosine decay")
    print(f"   Total steps: {total_steps}\n")

    return optimizer, scheduler
