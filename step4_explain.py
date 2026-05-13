"""
step4_explain.py — Step 4: Explainability (XAI)

This script loads the trained baseline model and runs Grad-CAM and
Attention Rollout on sample images to visualize what the model learned.
"""

import sys
import os
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Our internal configuration and modules
from src.config import DEVICE, NUM_CLASSES, MODEL_NAME, PROJECT_ROOT, CHECKPOINT_DIR, PLOT_DIR
from src.model import create_vit_model
from src.dataset import load_nih_metadata
from src.explainability import visualize_batch


def main():
    print("\n" + "🔍 " * 20)
    print("  STEP 4 — Explainability (XAI)")
    print("🔍 " * 20 + "\n")
    
    print(f"  🖥️  Device context : {DEVICE}")
    
    # ── 1. Select Sample Images ──
    print("  📂 Selecting 4 sample images from the dataset...")
    df = load_nih_metadata()
    
    # Filter by classes
    df_cons = df[df["class_name"] == "Consolidation"].reset_index(drop=True)
    df_nofind = df[df["class_name"] == "No Finding"].reset_index(drop=True)
    
    if len(df_cons) < 2 or len(df_nofind) < 2:
        print("❌ Not enough images found for the targeted classes.")
        return
        
    # Randomly select 2 images from each class
    seed = 42
    
    samples_cons = df_cons.sample(n=2, random_state=seed)
    samples_nofind = df_nofind.sample(n=2, random_state=seed)
    
    selected_df = pd.concat([samples_cons, samples_nofind])
    
    image_paths = selected_df["image_path"].tolist()
    true_labels = selected_df["class_name"].tolist()
    
    print("  📸 Chosen Samples:")
    for p, l in zip(image_paths, true_labels):
        print(f"      • {os.path.basename(p)}  ({l})")
        
    # ── 2. Evaluate Models ──
    model_files = ["best_model_real_only.pt", "best_model_hybrid.pt"]
    
    for model_file in model_files:
        ckpt_path = CHECKPOINT_DIR / model_file
        
        print("\n" + "-" * 60)
        print(f"  🚀 Evaluating Model: {model_file}")
        print("-" * 60)
        
        if not ckpt_path.exists():
            print(f"❌ Cannot find model checkpoint at {ckpt_path}. Skipping.")
            continue
            
        print(f"  📂 Loading checkpoint: {ckpt_path}")
        model = create_vit_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME, attn_implementation="eager")
        
        # 1. Hata almamak için modeli önce güvenli liman olan CPU'ya yükle
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        
        # 2. Ağırlıkları modele aktar
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint) # fallback
            
        # Force native eager attention for explainability
        model.config._attn_implementation = "eager"
        model.config.output_attentions = True
            
        # 3. Modeli şimdi ekran kartına (DirectML) gönder
        model.to(DEVICE)
        
        model.eval()
        print("  ✅ Model loaded successfully.")
            
        # ── 3. Generate Heatmaps ──
        print("  🎨 Generating Attention Heatmaps (Rollout & Grad-CAM)...")
        
        # Dynamically update save_dir
        save_dir = str(PLOT_DIR / "explainability" / model_file.replace(".pt", ""))
        os.makedirs(save_dir, exist_ok=True)
        
        # This function automatically handles moving data to DEVICE and saving the figures
        results = visualize_batch(
            model=model,
            image_paths=image_paths,
            true_labels=true_labels,
            save_dir=save_dir,
            methods=("rollout", "gradcam")
        )
        
        print(f"  ✅ Explanations saved to: {save_dir}")

    print("\n" + "=" * 60)
    print("  ✅ Step 4 Complete! All models evaluated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
