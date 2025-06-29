import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional

from .backbone import FeatureExtractor, Classifier, Discriminator, NonAdversarialDiscriminator
from .vae import ConvVAE
from utils.metrics import calculate_confidence, calculate_prototype_similarity, calculate_ukc_score, calculate_ukd_score


class DualPathOutlierEstimation:
    """Dual-path outlier estimation for unknown class detection."""
    
    def __init__(self, num_classes: int, feature_dim: int, update_rate: float = 0.9):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.update_rate = update_rate
        
        # Initialize prototypes for each class
        self.prototypes = torch.zeros(num_classes, feature_dim)
        self.prototype_counts = torch.zeros(num_classes)
        
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update class prototypes using exponential moving average."""
        for i in range(self.num_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                class_mean = class_features.mean(dim=0)
                
                if self.prototype_counts[i] == 0:
                    self.prototypes[i] = class_mean
                else:
                    self.prototypes[i] = (self.update_rate * self.prototypes[i] + 
                                        (1 - self.update_rate) * class_mean)
                
                self.prototype_counts[i] += class_mask.sum()
    
    def calculate_ukc_scores(self, features: torch.Tensor, logits: torch.Tensor, 
                           confidence_threshold: float = 0.8) -> torch.Tensor:
        """Calculate unknown class scores using dual-path estimation."""
        # Path 1: Confidence-based scoring
        probabilities = F.softmax(logits, dim=1)
        confidence = calculate_confidence(probabilities.detach().cpu().numpy())
        confidence = torch.from_numpy(confidence).to(features.device)
        
        # Path 2: Prototype-based scoring
        similarity = calculate_prototype_similarity(features.detach().cpu().numpy(), 
                                                  self.prototypes.cpu().numpy())
        similarity = torch.from_numpy(similarity).to(features.device)
        
        # Calculate UKC scores
        ukc_scores = calculate_ukc_score(confidence.cpu().numpy(), 
                                       similarity.cpu().numpy(), 
                                       confidence_threshold)
        ukc_scores = torch.from_numpy(ukc_scores).to(features.device)
        
        return ukc_scores


class USSLModel(nn.Module):
    """Universal Semi-Supervised Learning Model."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_classes = config['dataset']['num_classes']
        self.feature_dim = config['model']['feature_dim']
        self.classifier_dim = config['model']['classifier_dim']
        self.vae_latent_dim = config['model']['vae_latent_dim']
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=config['model']['backbone'],
            feature_dim=self.feature_dim,
            dropout_rate=config['model']['dropout_rate']
        )
        
        # Classifier
        self.classifier = Classifier(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            classifier_dim=self.classifier_dim,
            dropout_rate=config['model']['dropout_rate']
        )
        
        # VAE for domain detection
        self.vae = ConvVAE(
            input_channels=3,
            latent_dim=self.vae_latent_dim,
            image_size=config['dataset']['image_size'][0]
        )
        
        # Domain discriminators
        self.adversarial_discriminator = Discriminator(
            feature_dim=self.feature_dim
        )
        
        self.non_adversarial_discriminator = NonAdversarialDiscriminator(
            feature_dim=self.feature_dim
        )
        
        # Dual-path outlier estimation
        self.doe = DualPathOutlierEstimation(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            update_rate=config['training']['doe']['prototype_update_rate']
        )
        
        # Domain prototypes for unknown domain detection
        self.domain_prototypes = None
        self.domain_prototype_counts = None
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Extract features
        features = self.feature_extractor(x)
        
        # Classification
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        
        # VAE reconstruction
        vae_output = self.vae(x)
        
        # Domain discrimination
        domain_prob = self.non_adversarial_discriminator(features)
        
        return {
            'features': features,
            'logits': logits,
            'probabilities': probabilities,
            'vae_recon': vae_output['recon_x'],
            'vae_mu': vae_output['mu'],
            'vae_log_var': vae_output['log_var'],
            'domain_prob': domain_prob
        }
    
    def calculate_ukc_scores(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Calculate unknown class scores."""
        return self.doe.calculate_ukc_scores(
            features, logits, 
            self.config['training']['doe']['confidence_threshold']
        )
    
    def calculate_ukd_scores(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Calculate unknown domain scores."""
        # VAE reconstruction error
        vae_output = self.vae(x)
        recon_error = F.mse_loss(vae_output['recon_x'], x, reduction='none')
        recon_error = recon_error.mean(dim=[1, 2, 3])  # Average over spatial dimensions
        
        # Domain similarity (if domain prototypes are available)
        if self.domain_prototypes is not None:
            domain_similarity = calculate_prototype_similarity(
                features.detach().cpu().numpy(),
                self.domain_prototypes.cpu().numpy()
            )
            domain_similarity = torch.from_numpy(domain_similarity).to(features.device)
        else:
            domain_similarity = torch.zeros(features.size(0), 1).to(features.device)
        
        # Calculate UKD scores
        ukd_scores = calculate_ukd_score(
            recon_error.detach().cpu().numpy(),
            domain_similarity.detach().cpu().numpy()
        )
        ukd_scores = torch.from_numpy(ukd_scores).to(features.device)
        
        return ukd_scores
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update class prototypes."""
        self.doe.update_prototypes(features, labels)
    
    def update_domain_prototypes(self, features: torch.Tensor, domain_labels: torch.Tensor):
        """Update domain prototypes."""
        if self.domain_prototypes is None:
            num_domains = domain_labels.max().item() + 1
            self.domain_prototypes = torch.zeros(num_domains, self.feature_dim).to(features.device)
            self.domain_prototype_counts = torch.zeros(num_domains).to(features.device)
        
        for i in range(self.domain_prototypes.size(0)):
            domain_mask = (domain_labels == i)
            if domain_mask.sum() > 0:
                domain_features = features[domain_mask]
                domain_mean = domain_features.mean(dim=0)
                
                if self.domain_prototype_counts[i] == 0:
                    self.domain_prototypes[i] = domain_mean
                else:
                    self.domain_prototypes[i] = (0.9 * self.domain_prototypes[i] + 
                                               0.1 * domain_mean)
                
                self.domain_prototype_counts[i] += domain_mask.sum()


class USSLLoss:
    """Loss functions for USSL training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loss_weights = config['loss_weights']
        
    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Classification loss."""
        return F.cross_entropy(logits, labels)
    
    def consistency_loss(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """Consistency loss for SSL."""
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)
        return F.mse_loss(probs1, probs2)
    
    def entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy minimization loss."""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.mean()
    
    def vae_loss(self, vae_output: Dict[str, torch.Tensor], x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """VAE reconstruction and KL loss."""
        return self.vae.total_loss(
            vae_output['recon_x'], x,
            vae_output['mu'], vae_output['log_var'],
            beta=self.config['training']['ussl']['vae_weight']
        )
    
    def adversarial_loss(self, discriminator_output: torch.Tensor, 
                        is_source: bool) -> torch.Tensor:
        """Adversarial loss for domain adaptation."""
        if is_source:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        
        return F.binary_cross_entropy(discriminator_output, target)
    
    def total_loss(self, model_output: Dict[str, torch.Tensor], 
                   labels: torch.Tensor, x: torch.Tensor,
                   ukc_scores: torch.Tensor, ukd_scores: torch.Tensor,
                   consistency_logits: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate total loss."""
        losses = {}
        
        # Classification loss (only for labeled data)
        if labels.max() >= 0:  # Has valid labels
            losses['classification'] = self.classification_loss(
                model_output['logits'], labels
            )
        else:
            losses['classification'] = torch.tensor(0.0).to(x.device)
        
        # Consistency loss
        if consistency_logits is not None:
            losses['consistency'] = self.consistency_loss(
                model_output['logits'], consistency_logits
            )
        else:
            losses['consistency'] = torch.tensor(0.0).to(x.device)
        
        # Entropy loss
        losses['entropy'] = self.entropy_loss(model_output['logits'])
        
        # VAE loss
        vae_loss, vae_losses = self.vae_loss(model_output, x)
        losses['vae_reconstruction'] = vae_losses['recon_loss']
        losses['vae_kl'] = vae_losses['kl_loss']
        
        # Weighted combination
        total_loss = (
            self.loss_weights['classification'] * losses['classification'] +
            self.loss_weights['consistency'] * losses['consistency'] +
            self.loss_weights['entropy'] * losses['entropy'] +
            self.loss_weights['vae_reconstruction'] * losses['vae_reconstruction'] +
            self.loss_weights['vae_kl'] * losses['vae_kl']
        )
        
        # Add UKC and UKD weighted losses
        if ukc_scores.sum() > 0:
            ukc_weighted_loss = (ukc_scores * losses['entropy']).mean()
            total_loss += self.config['training']['ussl']['lambda_ukc'] * ukc_weighted_loss
            losses['ukc_weighted'] = ukc_weighted_loss
        
        if ukd_scores.sum() > 0:
            ukd_weighted_loss = (ukd_scores * losses['entropy']).mean()
            total_loss += self.config['training']['ussl']['lambda_ukd'] * ukd_weighted_loss
            losses['ukd_weighted'] = ukd_weighted_loss
        
        losses['total'] = total_loss
        
        return total_loss, losses 