# =========================================
# 0. Imports
# =========================================
import json
import os
import pathlib
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import Config
from dataset_utils import FusedDataset, prepare_fused_samples
from engine import train_one_epoch, validate
from model import build_model
from plotting_utils import LivePlot

# =========================================
# 1. Helper Functions
# =========================================
def set_seed(seed):
    """Sets random seeds for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def auto_detect_and_prepare_datasets(root_dir):
    """Scans a root directory for datasets, unzipping any .zip files if necessary."""
    print(f"🔍 Scanning for datasets in: {root_dir}")
    datasets_root_path = pathlib.Path(root_dir)
    valid_dataset_paths = []
    if not datasets_root_path.is_dir():
        print(f"❌ Error: Root datasets directory not found at '{root_dir}'")
        return []
    for item_path in datasets_root_path.iterdir():
        if item_path.is_file() and item_path.suffix == '.zip':
            extracted_folder_path = datasets_root_path / item_path.stem
            if not extracted_folder_path.is_dir():
                print(f"  - Found ZIP file: '{item_path.name}'. Extracting...")
                with zipfile.ZipFile(item_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder_path)
                print(f"  ✔ Extracted to: '{extracted_folder_path.name}'")
            else:
                print(f"  - Found ZIP file: '{item_path.name}'. Matching folder already exists.")
            valid_dataset_paths.append(str(extracted_folder_path))
        elif item_path.is_dir():
            print(f"  - Found dataset folder: '{item_path.name}'")
            valid_dataset_paths.append(str(item_path))
    print(f"\n✅ Finished detection. Using {len(valid_dataset_paths)} dataset(s): {valid_dataset_paths}\n")
    return valid_dataset_paths

# =========================================
# 2. Data Preparation Function
# =========================================
def prepare_data(cfg):
    """Fuses datasets, defines transforms, and creates DataLoaders using a config object."""
    dataset_directories = auto_detect_and_prepare_datasets(cfg.DATASETS_ROOT_DIR)
    train_samples, valid_samples, unified_class_map, class_weights_list = prepare_fused_samples(
        dataset_dirs=dataset_directories, 
        selection_mode=cfg.CLASS_SELECTION_MODE,
        top_n=cfg.TOP_N_CLASSES,
        specific_classes=cfg.SPECIFIC_CLASSES
    )
    if not train_samples: raise ValueError("Dataset preparation resulted in no training samples.")
    num_classes = len(unified_class_map)
    with open(cfg.CLASS_MAP_PATH, "w") as f: json.dump(unified_class_map, f, indent=4)
    print(f"Saved class mapping for {num_classes} classes to {cfg.CLASS_MAP_PATH}")

    class_weights = torch.tensor(class_weights_list, dtype=torch.float).to(cfg.DEVICE)
    print(f"Calculated weights: {class_weights}")

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
    
    if not valid_samples:
        print("No pre-split validation set found. Performing 80/20 random split on the list of training samples.")
        train_size = int(0.8 * len(train_samples))
        val_size = len(train_samples) - train_size
        generator = torch.Generator().manual_seed(cfg.RANDOM_SEED)
        train_indices, val_indices = random_split(range(len(train_samples)), [train_size, val_size], generator=generator)
        final_train_samples = [train_samples[i] for i in train_indices]
        final_valid_samples = [train_samples[i] for i in val_indices]
    else:
        print("Using pre-split validation set found in dataset folders.")
        final_train_samples = train_samples
        final_valid_samples = valid_samples

    train_dataset = FusedDataset(final_train_samples, transform=train_transform, preload_into_ram=cfg.PRELOAD_DATASET_INTO_RAM, safety_margin=cfg.MEMORY_SAFETY_MARGIN)
    val_dataset = FusedDataset(final_valid_samples, transform=val_transform, preload_into_ram=cfg.PRELOAD_DATASET_INTO_RAM, safety_margin=cfg.MEMORY_SAFETY_MARGIN)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, num_classes, class_weights

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

    train_loader, val_loader, num_classes, class_weights = prepare_data(cfg)
    
    model = build_model(num_classes, cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    live_plot = LivePlot(save_path=cfg.PLOT_SAVE_PATH)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    for epoch in range(cfg.EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())
        live_plot.update(history)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Time: {elapsed_time:.1f}s")
        
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"Model saved. Validation loss improved to {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.PATIENCE:
                print("Early stopping.")
                break
                
    live_plot.save_and_close()