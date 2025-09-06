import torch
from tqdm import tqdm
import random
import numpy as np

def mixup_data(x, y, device, alpha=1.0):
    """Applies Mixup augmentation to a batch of data."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, device, alpha=1.0):
    """Applies CutMix augmentation to a batch of data."""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0)).to(device)
    W, H = x.size()[-1], x.size()[-2]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[rand_index], lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the mixed loss for Mixup/Cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        rand_val = random.random()
        
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            if rand_val < 0.25:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, device)
                outputs = model(inputs)
                loss = mix_criterion(criterion, outputs, y_a, y_b, lam)
            elif rand_val < 0.5:
                inputs, y_a, y_b, lam = cutmix_data(inputs, labels, device)
                outputs = model(inputs)
                loss = mix_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    """Runs a single validation epoch."""
    model.eval()
    val_loss = 0.0
    val_corrects = torch.tensor(0.0).to(device)
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels)
            
    return val_loss / len(loader.dataset), val_corrects.double() / len(loader.dataset)
