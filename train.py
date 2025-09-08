# =========================================
# 0. Imports
# =========================================
import json # Imports the JSON module for working with JSON files (e.g., saving the class map).
import os # Imports the OS module for interacting with the operating system (not directly used here but good practice).
import pathlib # Imports the pathlib module for an object-oriented way to handle filesystem paths.
import random # Imports the random module for generating random numbers (used for augmentations).
import time # Imports the time module for timing operations like epoch duration.
import zipfile # Imports the zipfile module for working with ZIP archives.

import numpy as np # Imports NumPy for numerical operations, especially for data augmentation calculations.
import torch # Imports the main PyTorch library.
import torch.nn as nn # Imports PyTorch's neural network module for layers and loss functions.
import torch.optim as optim # Imports PyTorch's optimization module for optimizers like Adam.
from torch.utils.data import DataLoader, random_split # Imports DataLoader for batching and random_split for creating validation sets.
from torchvision import transforms # Imports torchvision's transforms module for common image transformations.

from config import Config # Imports the Config class from our local config.py file.
from dataset_utils import FusedDataset, prepare_fused_samples # Remove ChunkedFusedDataset import
from engine import train_one_epoch, validate # Imports our core training and validation functions from engine.py.
from model import build_model # Imports our model-building function from model.py.
from plotting_utils import KFoldTrainingPlotter
from sklearn.model_selection import StratifiedKFold

# =========================================
# 1. Helper Functions
# =========================================
def set_seed(seed):
    """Sets random seeds for all relevant libraries to ensure reproducibility."""
    random.seed(seed) # Sets the seed for Python's built-in random module.
    np.random.seed(seed) # Sets the seed for NumPy's random number generator.
    torch.manual_seed(seed) # Sets the seed for PyTorch's CPU operations.
    if torch.cuda.is_available(): # Checks if a CUDA-enabled GPU is available.
        torch.cuda.manual_seed_all(seed) # Sets the seed for all available GPUs.
        torch.backends.cudnn.benchmark = True # Enables cuDNN's auto-tuner, which can speed up training if input sizes don't change.

# =========================================
# 2. Data Preparation Function
# =========================================
def prepare_all_samples(cfg):
    """Fuses datasets and returns a single list of all available samples."""
    # The list of dataset directories now comes directly from the config.
    train_samples, valid_samples, unified_class_map, _ = prepare_fused_samples(
        dataset_dirs=cfg.DATASET_DIRECTORIES, 
        selection_mode=cfg.CLASS_SELECTION_MODE,
        top_n=cfg.TOP_N_CLASSES,
        specific_classes=cfg.SPECIFIC_CLASSES
    )
    if not train_samples: 
        raise ValueError("Dataset preparation resulted in no training samples.")
        
    # Combine pre-split train and validation samples into one list for K-Fold
    all_samples = train_samples + (valid_samples if valid_samples else [])
    
    num_classes = len(unified_class_map)
    with open(cfg.CLASS_MAP_PATH, "w") as f: 
        json.dump(unified_class_map, f, indent=4)
    print(f"Saved class mapping for {num_classes} classes to {cfg.CLASS_MAP_PATH}")

    return all_samples, unified_class_map

# =========================================
# 3. Main Execution Block
# =========================================
if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)
    
    if torch.cuda.is_available():
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No GPU found. Training will run on CPU.")
    print(f"Using device: {cfg.DEVICE}")

    # --- Get All Samples for K-Fold ---
    all_samples, unified_class_map = prepare_all_samples(cfg)
    num_classes = len(unified_class_map)
    
    # Extract labels for stratified splitting
    all_labels = [label for _, label in all_samples]

    # --- Define Transforms ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- K-Fold Cross-Validation Setup ---
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED)
    all_histories = []
    fold_metrics = []

    # =========================================
    # 4. Main K-Fold Training Loop
    # =========================================
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, all_labels)):
        print(f"\n{'='*20} FOLD {fold+1}/{N_SPLITS} {'='*20}")

        # --- Create fold-specific samples and DataLoaders ---
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        train_dataset = FusedDataset(train_samples, transform=train_transform, preload=cfg.PRELOAD_DATASET_INTO_RAM, safety_margin=cfg.MEMORY_SAFETY_MARGIN)
        num_workers = 4 if not train_dataset.samples_are_preloaded else 0
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)

        val_dataset = FusedDataset(val_samples, transform=val_transform, preload=True, safety_margin=cfg.MEMORY_SAFETY_MARGIN)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # --- Calculate Class Weights for the current fold ---
        from collections import Counter
        label_counts = Counter([label for _, label in train_samples])
        total_samples = len(train_samples)
        class_weights = torch.zeros(num_classes)
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            class_weights[i] = total_samples / (num_classes * count) if count > 0 else 1.0
        class_weights = class_weights.to(cfg.DEVICE)

        # --- Re-initialize Model, Optimizer, etc. for each fold ---
        model = build_model(num_classes, cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, verbose=True)
        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # --- Fold-specific Training History ---
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0
        
        # --- Inner Training Loop for the current fold ---
        for epoch in range(cfg.EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg.DEVICE)
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, cfg.DEVICE)
            
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)
            
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed_time:.1f}s")
            
            if val_loss < best_val_loss:
                fold_model_path = f"{cfg.MODEL_SAVE_PATH.rsplit('.', 1)[0]}_fold{fold+1}.pth"
                torch.save(model.state_dict(), fold_model_path)
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(f"Model for fold {fold+1} saved. Validation loss improved to {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break
        
        all_histories.append(history)
        fold_metrics.append({
            'best_val_loss': best_val_loss,
            'final_val_acc': history['val_acc'][-1],
            'final_val_f1': history['val_f1'][-1]
        })

    # =========================================
    # 5. Final Evaluation and Plotting
    # =========================================
    print(f"\n{'='*20} K-FOLD SUMMARY {'='*20}")
    avg_val_loss = np.mean([m['best_val_loss'] for m in fold_metrics])
    avg_val_acc = np.mean([m['final_val_acc'] for m in fold_metrics])
    avg_val_f1 = np.mean([m['final_val_f1'] for m in fold_metrics])

    print(f"Average Best Validation Loss across {N_SPLITS} folds: {avg_val_loss:.4f}")
    print(f"Average Final Validation Accuracy across {N_SPLITS} folds: {avg_val_acc:.4f}")
    print(f"Average Final Validation F1-Score across {N_SPLITS} folds: {avg_val_f1:.4f}")

    plotter = KFoldTrainingPlotter(save_path=cfg.PLOT_SAVE_PATH)
    plotter.plot_and_save(all_histories)