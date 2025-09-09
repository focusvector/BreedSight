import torch.nn as nn
from torchvision import models

class BreedSightModel(nn.Module):
    """A custom model using a pre-trained InceptionV3 backbone for fine-tuning."""
    def __init__(self, num_classes):
        """Initializes the model architecture."""
        super().__init__()
        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # 1. Freeze all backbone parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Only unfreeze classifier
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        # 3. Replace the final classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # 4. Disable the auxiliary output
        if hasattr(self.backbone, 'AuxLogits'):
            self.backbone.AuxLogits = nn.Identity()

    def forward(self, x):
        """Defines the forward pass of the model."""
        out = self.backbone(x)
        if hasattr(out, 'logits'):
            return out.logits
        return out

def build_model(num_classes, device):
    """A helper function to create the model and move it to the correct device."""
    model = BreedSightModel(num_classes=num_classes)
    return model.to(device)
