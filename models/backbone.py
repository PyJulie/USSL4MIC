import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Optional


class FeatureExtractor(nn.Module):
    """Feature extractor based on pre-trained backbones."""
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, 
                 feature_dim: int = 2048, dropout_rate: float = 0.5):
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load pre-trained backbone
        if backbone.startswith('resnet'):
            self.backbone = self._get_resnet(backbone, pretrained)
        elif backbone.startswith('efficientnet'):
            self.backbone = self._get_efficientnet(backbone, pretrained)
        elif backbone.startswith('densenet'):
            self.backbone = self._get_densenet(backbone, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
    
    def _get_resnet(self, backbone: str, pretrained: bool):
        """Get ResNet backbone."""
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model.num_features = 512
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            model.num_features = 512
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.num_features = 2048
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model.num_features = 2048
        elif backbone == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
            model.num_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet: {backbone}")
        
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-2])
        return model
    
    def _get_efficientnet(self, backbone: str, pretrained: bool):
        """Get EfficientNet backbone."""
        model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        # Get the last layer's output dimension
        model.num_features = model.feature_info.channels()[-1]
        return model
    
    def _get_densenet(self, backbone: str, pretrained: bool):
        """Get DenseNet backbone."""
        if backbone == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            model.num_features = 1024
        elif backbone == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
            model.num_features = 1664
        elif backbone == 'densenet201':
            model = models.densenet201(pretrained=pretrained)
            model.num_features = 1920
        else:
            raise ValueError(f"Unsupported DenseNet: {backbone}")
        
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        features = self.backbone(x)
        
        # Handle different backbone outputs
        if self.backbone_name.startswith('efficientnet'):
            # EfficientNet returns a list of features
            features = features[-1]
        
        # Project features to desired dimension
        projected_features = self.feature_projection(features)
        
        return projected_features


class Classifier(nn.Module):
    """Multi-class classifier."""
    
    def __init__(self, feature_dim: int, num_classes: int, classifier_dim: int = 512, 
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim, classifier_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify features."""
        return self.classifier(features)


class Discriminator(nn.Module):
    """Domain discriminator for adversarial training."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Discriminate between source and target domains."""
        return self.discriminator(features)


class NonAdversarialDiscriminator(nn.Module):
    """Non-adversarial discriminator for domain detection."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict domain probability."""
        return self.discriminator(features) 