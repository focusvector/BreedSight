import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Runs a single training epoch, calculating loss and accuracy.
    This version uses a standard training loop without Mixup or Cutmix
    to avoid conflicts with TrivialAugment.
    """
    model.train()
    running_loss = 0.0
    total_corrects = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        # Skip batch if it's empty (can happen with collate_fn filtering)
        if inputs is None or labels is None:
            continue
            
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        # Standard training step
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy for every batch
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = (total_corrects / total_samples) if total_samples > 0 else 0.0
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Runs a single validation epoch and calculates loss, accuracy, and other metrics."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            # Skip batch if it's empty
            if inputs is None or labels is None:
                continue

            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics from collected predictions and labels
    avg_loss = val_loss / total_samples if total_samples > 0 else 0.0
    
    # Ensure there's something to calculate metrics on
    if not all_labels or not all_preds:
        return avg_loss, 0.0, 0.0, 0.0, 0.0

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Use zero_division=0 to avoid warnings when a class has no predictions
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1
