import torch
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# --- MixUp helpers ---
def _mixup(inputs, labels, alpha=0.2, device=None):
    if alpha <= 0.0:
        return inputs, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = float(np.clip(lam, 0.0, 1.0))
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device if device is None else device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    y_a, y_b = labels, labels[index]
    return mixed_inputs, y_a, y_b, lam

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, mixup_prob=0.0, mixup_alpha=0.2):
    """
    Runs a single training epoch, calculating loss and accuracy.
    Optional MixUp for light regularization.
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
        use_mixup = (mixup_prob > 0.0) and (np.random.rand() < mixup_prob)
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            if use_mixup:
                mixed_inputs, y_a, y_b, lam = _mixup(inputs, labels, alpha=mixup_alpha, device=device)
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                total_corrects += torch.sum(preds == labels).item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
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
        return avg_loss, 0.0, 0.0, 0.0, 0.0, None

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Use zero_division=0 to avoid warnings when a class has no predictions
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, cm
