import torch
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def mixup_data(x, y, device, alpha=1.0):
    """Applies Mixup augmentation to a batch of data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generates a random bounding box for Cutmix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform distribution for center of the patch
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, device, alpha=1.0):
    """Applies Cutmix augmentation to a batch of data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    lam = np.clip(lam, 0, 1)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the mixed loss for Mixup/Cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Runs a single training epoch and calculates loss and accuracy."""
    model.train()
    running_loss = 0.0
    total_corrects = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        rand_val = random.random()
        
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            if rand_val < 0.5: # Using Mixup/Cutmix
                use_mix = rand_val < 0.25
                if use_mix:
                    mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, device)
                else:
                    mixed_inputs, y_a, y_b, lam = cutmix_data(inputs, labels, device)
                
                outputs = model(mixed_inputs)
                loss = mix_criterion(criterion, outputs, y_a, y_b, lam)
                # Accuracy is not calculated for mixed batches as labels are blended
            else: # Standard training
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
        
    epoch_loss = running_loss / len(loader.dataset)
    # Calculate accuracy only on non-augmented samples
    epoch_acc = (total_corrects / total_samples) if total_samples > 0 else 0.0
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Runs a single validation epoch and calculates loss, accuracy, and other metrics."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics from collected predictions and labels
    avg_loss = val_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Use zero_division=0 to avoid warnings when a class has no predictions
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1
