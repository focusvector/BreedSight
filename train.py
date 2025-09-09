# =========================================
# 0. Imports
# =========================================
import json # Imports the JSON module for working with JSON files (e.g., saving the class map).
import os # Imports the OS m                        cm_save_path = f"{cfg.MODEL_SAVE_PATH.rsplit('.', 1)[0]}_fold{fold+1}_confusion.png"dule for interacting with the operating system (not directly used here but good practice).
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

def collate_fn(batch):
    """A custom collate function that filters out None values.
    This is used to handle corrupted images that are skipped by the FusedDataset.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

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
# 2.b Playbook: staged fine-tuning adjustments
# =========================================
def adjust_training(cfg, model, optimizer, scheduler, epoch, val_loss_history):
    """Adjust training per playbook based on epoch and plateau.
    Returns possibly-updated (model, optimizer, scheduler) and a message string if changed.
    """
    change_msgs = []

    # Phase switch at UNFREEZE_EPOCH (1-based)
    if epoch == (cfg.UNFREEZE_EPOCH - 1):
        # Unfreeze deeper blocks
        for name, child in model.backbone.named_children():
            if name in cfg.UNFREEZE_LAYERS:
                for p in child.parameters():
                    p.requires_grad = True
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=cfg.FINETUNE_LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.EPOCHS - epoch), eta_min=1e-6)
        change_msgs.append(f"[Adjust] Epoch {epoch+1}: Unfroze {cfg.UNFREEZE_LAYERS}; LR -> {cfg.FINETUNE_LR}")

    # Plateau detection over 5 epochs: compare last 5 vs best before
    plateau_5 = False
    plateau_10 = False
    plateau_15 = False
    n = len(val_loss_history)
    if n >= 6:
        recent5 = val_loss_history[-5:]
        best_before = min(val_loss_history[:-5]) if n > 5 else float('inf')
        if min(recent5) >= best_before:
            plateau_5 = True
    if n >= 11:
        recent10 = val_loss_history[-10:]
        best_before10 = min(val_loss_history[:-10]) if n > 10 else float('inf')
        if min(recent10) >= best_before10:
            plateau_10 = True
    if n >= 16:
        recent15 = val_loss_history[-15:]
        best_before15 = min(val_loss_history[:-15]) if n > 15 else float('inf')
        if min(recent15) >= best_before15:
            plateau_15 = True

    # a) If plateau >=5 epochs: lower LR to 1e-4
    if plateau_5:
        for g in optimizer.param_groups:
            if g['lr'] > 1e-4:
                g['lr'] = 1e-4
                change_msgs.append(f"[Adjust] Epoch {epoch+1}: Plateau detected; LR -> 1e-4")

    # b) If plateau >=10: stronger regularization
    if plateau_10:
        if getattr(cfg, 'DROPOUT_RATE', 0.3) < 0.4:
            cfg.DROPOUT_RATE = 0.4
            change_msgs.append(f"[Adjust] Epoch {epoch+1}: Dropout -> 0.4")
        if getattr(cfg, 'RANDOM_ERASE_P', 0.0) < 0.2:
            cfg.RANDOM_ERASE_P = 0.2
            change_msgs.append(f"[Adjust] Epoch {epoch+1}: RandomErasing p -> 0.2")
        if getattr(cfg, 'MIXUP_PROB', 0.0) < 0.1:
            cfg.MIXUP_PROB = 0.1
            change_msgs.append(f"[Adjust] Epoch {epoch+1}: MixUp prob -> 0.1 (alpha={cfg.MIXUP_ALPHA})")

    # c) If plateau >=15: unfreeze Mixed_5d/e and set LR=5e-5
    if plateau_15:
        for name, child in model.backbone.named_children():
            if name in ["Mixed_5d", "Mixed_5e"]:
                for p in child.parameters():
                    p.requires_grad = True
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=5e-5, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.EPOCHS - epoch), eta_min=1e-6)
        change_msgs.append(f"[Adjust] Epoch {epoch+1}: Unfroze Mixed_5d/e; LR -> 5e-5")

    return model, optimizer, scheduler, change_msgs

# =========================================
# 3. Main Execution Block
# =========================================
if __name__ == "__main__":
    cfg = Config()
    set_seed(42) # Set a fixed random seed for reproducibility
    
    # Enable anti-overfitting and fine-tuning from the start
    cfg.DROPOUT_RATE = 0.4
    cfg.RANDOM_ERASE_P = 0.2
    cfg.MIXUP_PROB = 0.1
    cfg.MIXUP_ALPHA = getattr(cfg, 'MIXUP_ALPHA', 0.2)

    # --- Checkpoint Setup ---
    CHECKPOINT_PATH = "training_checkpoint.pth"
    start_fold = 0
    start_epoch = 0
    all_histories = []
    fold_metrics = []
    
    if torch.cuda.is_available():
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No GPU found. Training will run on CPU.")
    print(f"Using device: {cfg.DEVICE}")

    # --- Load Checkpoint if it exists ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch']
        all_histories = checkpoint['all_histories']
        fold_metrics = checkpoint['fold_metrics']
        # If the last saved epoch was the final one, we start the next fold from epoch 0
        if start_epoch == cfg.EPOCHS:
            start_fold += 1
            start_epoch = 0
            
    # --- Get All Samples for K-Fold ---
    all_samples, unified_class_map = prepare_all_samples(cfg)
    num_classes = len(unified_class_map)
    
    # Extract labels for stratified splitting
    all_labels = [label for _, label in all_samples]

    # --- Define Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=getattr(cfg, 'RANDOM_ERASE_P', 0.0), value='random', inplace=False),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- K-Fold Cross-Validation Setup ---
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED)
    # Note: all_histories and fold_metrics are now initialized before this loop

    # =========================================
    # 4. Main K-Fold Training Loop
    # =========================================
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, all_labels)):
        # --- Skip completed folds if resuming ---
        if fold < start_fold:
            print(f"⏭️ Skipping completed fold {fold+1}/{N_SPLITS}")
            continue

        print(f"\n{'='*20} FOLD {fold+1}/{N_SPLITS} {'='*20}")

        # --- Create fold-specific samples and DataLoaders ---
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        train_dataset = FusedDataset(train_samples, transform=train_transform, preload=cfg.PRELOAD_DATASET_INTO_RAM, safety_margin=cfg.MEMORY_SAFETY_MARGIN)
        num_workers = 4 if not train_dataset.samples_are_preloaded else 0
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

        val_dataset = FusedDataset(val_samples, transform=val_transform, preload=True, safety_margin=cfg.MEMORY_SAFETY_MARGIN)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

        # --- Calculate Class Weights for the current fold ---
        from collections import Counter
        label_counts = Counter([label for _, label in train_samples])
        total_samples = len(train_samples)
        class_weights = torch.zeros(num_classes)

        for i in range(num_classes):
            count = label_counts.get(i, 0)
            class_weights[i] = total_samples / (num_classes * count) if count > 0 else 1.0
        class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
        # normalize to mean 1.0 to stabilize CE loss scale
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(cfg.DEVICE)
        print(f"Class weights: {class_weights}")

        # --- Re-initialize Model, Optimizer, etc. for each fold ---
        model = build_model(num_classes, cfg.DEVICE)
        
        # Unfreeze deeper Inception blocks from the start (Mixed_5d/e and Mixed_6a..Mixed_7c)
        for name, child in model.backbone.named_children():
            if name in cfg.UNFREEZE_LAYERS:
                for p in child.parameters():
                    p.requires_grad = True
        
        # Apply initial dropout rate across model
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = cfg.DROPOUT_RATE
        
        # Loss: optionally disable class weights, no label smoothing to pair with MixUp
        use_weights = getattr(cfg, 'USE_CLASS_WEIGHTS', True)
        criterion = nn.CrossEntropyLoss(weight=(class_weights if use_weights else None), label_smoothing=0.0)
        
        # Optimizer: fine-tune from the start using FINETUNE_LR
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=cfg.FINETUNE_LR, weight_decay=cfg.WEIGHT_DECAY)
        
        # Scheduler: warmup then cosine
        warmup_epochs = getattr(cfg, 'WARMUP_EPOCHS', 0)
        if warmup_epochs and warmup_epochs > 0:
            warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: min((ep + 1) / warmup_epochs, 1.0))
            cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.EPOCHS - warmup_epochs, 1), eta_min=1e-6)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
        
        # Diagnostics: print trainable parameter groups
        trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        print("Trainable layers and param counts:")
        for n, cnt in trainable:
            print(n, cnt)
        print("Total trainable params:", sum(cnt for _, cnt in trainable))
        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # --- Fold-specific Training History ---
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        # --- Load state from checkpoint if resuming this fold ---
        if fold == start_fold and start_epoch > 0:
            print(f"Loading model and optimizer state for fold {fold+1} from epoch {start_epoch}")
            model.load_state_dict(checkpoint['model_state_dict'])
            # Ensure unfreezing from the start (keep consistent with current policy)
            for name, child in model.backbone.named_children():
                if name in cfg.UNFREEZE_LAYERS:
                    for p in child.parameters():
                        p.requires_grad = True
            params_to_update = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params_to_update, lr=cfg.FINETUNE_LR, weight_decay=cfg.WEIGHT_DECAY)
            # Rebuild warmup+cosine scheduler with offset
            warmup_epochs = getattr(cfg, 'WARMUP_EPOCHS', 0)
            if warmup_epochs and warmup_epochs > 0:
                warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: min((ep + 1) / warmup_epochs, 1.0))
                cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.EPOCHS - warmup_epochs, 1), eta_min=1e-6)
                scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs], last_epoch=start_epoch-1)
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6, last_epoch=start_epoch-1)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            best_val_loss = checkpoint['best_val_loss']
            epochs_no_improve = checkpoint['epochs_no_improve']
            history = all_histories[-1] # The last history is the one we are resuming

        # --- Inner Training Loop for the current fold ---
        for epoch in range(start_epoch, cfg.EPOCHS):
            start_time = time.time()
            # Start with fixed settings; ensure current dropout/erasing settings are applied
            target_drop = getattr(cfg, 'DROPOUT_RATE', 0.4)
            for mod in model.modules():
                if isinstance(mod, nn.Dropout) and abs(mod.p - target_drop) > 1e-6:
                    mod.p = target_drop
            if hasattr(train_dataset, 'transform') and isinstance(train_dataset.transform, transforms.Compose):
                for t in getattr(train_dataset.transform, 'transforms', []):
                    if isinstance(t, transforms.RandomErasing) and abs(t.p - getattr(cfg, 'RANDOM_ERASE_P', 0.2)) > 1e-6:
                        t.p = cfg.RANDOM_ERASE_P

            # Report current LR
            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, cfg.DEVICE,
                mixup_prob=getattr(cfg, 'MIXUP_PROB', 0.2), mixup_alpha=getattr(cfg, 'MIXUP_ALPHA', 0.2)
            )
            val_loss, val_acc, val_precision, val_recall, val_f1, cm = validate(model, val_loader, criterion, cfg.DEVICE)
            
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)
            
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed_time:.1f}s")
            
            if val_loss < best_val_loss:
                fold_model_path = f"{cfg.MODEL_SAVE_PATH.rsplit('.', 1)[0]}_fold{fold+1}.pth"
                torch.save(model.state_dict(), fold_model_path)
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(f"Model for fold {fold+1} saved. Validation loss improved to {best_val_loss:.4f}")

                # save confusion matrix heatmap for this fold if available
                try:
                    # 'cm' should be available from validate return
                    cm_save_path = f"{cfg.MODEL_SAVE_PATH.rsplit('.', 1)[0]}_fold{fold+1}_confusion.png"
                    class_names = [k for k,v in sorted(unified_class_map.items(), key=lambda item: item[1])]
                    plotter = KFoldTrainingPlotter(save_path=cfg.PLOT_SAVE_PATH)
                    plotter.save_confusion_matrix(cm, class_names, cm_save_path)
                except Exception as e:
                    print(f"⚠️ Warning: Could not save confusion matrix for fold {fold+1}. Error: {e}")

                # --- Save Checkpoint ---
                # We save the history for the current fold before it's appended to all_histories
                temp_histories = all_histories + [history]
                torch.save({
                    'fold': fold,
                    'epoch': epoch + 1, # Save the next epoch to start from
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve,
                    'all_histories': temp_histories,
                    'fold_metrics': fold_metrics,
                }, CHECKPOINT_PATH)
                print(f"💾 Checkpoint saved at epoch {epoch+1} for fold {fold+1}")

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break
        
        # --- After each fold completes ---
        was_resumed = fold == start_fold and 'model_state_dict' in locals().get('checkpoint', {})
        if fold == start_fold: # If we were resuming, subsequent folds start from epoch 0
            start_epoch = 0

        # Only append history if the fold was actually run
        if not (fold < start_fold):
            if was_resumed:
                # If we resumed this fold, replace the placeholder history with the completed one
                all_histories[-1] = history
            else:
                 all_histories.append(history)

            fold_metrics.append({
                'best_val_loss': best_val_loss,
                'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
                'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0
            })

    # =========================================
    # 5. Final Evaluation and Plotting
    # =========================================
    print(f"\n{'='*20} K-FOLD SUMMARY {'='*20}")
    if fold_metrics:
        avg_val_loss = np.mean([m['best_val_loss'] for m in fold_metrics])
        avg_val_acc = np.mean([m['final_val_acc'] for m in fold_metrics])
        avg_val_f1 = np.mean([m['final_val_f1'] for m in fold_metrics])

        print(f"Average Best Validation Loss across {N_SPLITS} folds: {avg_val_loss:.4f}")
        print(f"Average Final Validation Accuracy across {N_SPLITS} folds: {avg_val_acc:.4f}")
        print(f"Average Final Validation F1-Score across {N_SPLITS} folds: {avg_val_f1:.4f}")
    else:
        print("No metrics to report. Training may not have completed any folds.")

    plotter = KFoldTrainingPlotter(save_path=cfg.PLOT_SAVE_PATH)
    plotter.plot_and_save(all_histories)

    # --- Clean up checkpoint on successful completion ---
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"\n✅ Training complete. Checkpoint file '{CHECKPOINT_PATH}' removed.")