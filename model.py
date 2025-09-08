import torch.nn as nn
from torchvision import models

def build_model(num_classes, device):
    """
    Builds an EfficientNet-B0 model with a custom classifier head.
    This model is lighter and often performs better on smaller datasets than InceptionV3.
    """
    # 1. Load pre-trained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 2. Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze more layers for deeper fine-tuning
    # Unfreezing from block 4 onwards to give the model more capacity to adapt.
    for i in range(4, 8): # Unfreezing features[4] through features[7]
        for param in model.features[i].parameters():
            param.requires_grad = True

    # 4. Replace the classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), # Use a dropout rate suitable for the smaller head
        nn.Linear(num_features, num_classes)
    )
    
    return model.to(device)
