import torch.nn as nn
from torchvision import models

class BreedSightModel(nn.Module):
    """A custom model using a pre-trained InceptionV3 backbone for fine-tuning."""
    def __init__(self, num_classes):
        """Initializes the model architecture."""
        super().__init__()
        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for name, child in self.backbone.named_children():
            if name in ['Mixed_7a', 'Mixed_7b', 'Mixed_7c', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True

        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )
        
        if hasattr(self.backbone, 'AuxLogits'):
            self.backbone.AuxLogits = nn.Identity()

    def forward(self, x):
        """Defines the forward pass of the model."""
        if self.training:
            outputs, _ = self.backbone(x)
            return outputs
        else:
            return self.backbone(x)

def build_model(num_classes, device):
    """A helper function to create the model and move it to the correct device."""
    model = BreedSightModel(num_classes=num_classes)
    return model.to(device)
