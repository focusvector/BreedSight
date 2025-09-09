import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def rand_bbox(size, lam):
    """Generate random bbox for CutMix.
    size: tensor shape (B, C, H, W)
    returns x1, y1, x2, y2 (ints)
    """
    _, _, H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix_data(x, y, device, alpha=1.0):
    """Apply CutMix on a batch and return mixed inputs and labels.
    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    lam = float(np.clip(lam, 0.0, 1.0))
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    x2 = x.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # Note: slicing order is [N, C, H, W]
    x2[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # adjust lambda to exactly match pixel ratio
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1.0 - area / float(x.size(2) * x.size(3))

    y_a, y_b = y, y[index]
    return x2, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed labels."""
    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Runs a single training epoch, calculating loss and accuracy.
    This version optionally applies CutMix augmentation to a portion of batches.
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
        
        # decide whether to apply CutMix for this batch
        do_cutmix = np.random.rand() < CUTMIX_PROB
        
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            if do_cutmix:
                mixed_inputs, y_a, y_b, lam = cutmix_data(inputs, labels, device, alpha=CUTMIX_ALPHA)
                outputs = model(mixed_inputs)
                loss = mix_criterion(criterion, outputs, y_a, y_b, lam)
                # do not count accuracy for mixed labels
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
        return avg_loss, 0.0, 0.0, 0.0, 0.0, None

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Use zero_division=0 to avoid warnings when a class has no predictions
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, cm

CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0
