import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VAE(nn.Module):
    """Variational AutoEncoder for domain detection."""
    
    def __init__(self, input_dim: int = 3 * 224 * 224, latent_dim: int = 128, 
                 hidden_dims: list = [512, 256, 128]):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.Module:
        """Build encoder network."""
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        layers = []
        hidden_dims_reversed = list(reversed(self.hidden_dims))
        in_dim = self.latent_dim
        
        for hidden_dim in hidden_dims_reversed:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        # Final reconstruction layer
        layers.append(nn.Linear(hidden_dims_reversed[-1], self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        x = x.view(x.size(0), -1)  # Flatten
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        
        return {
            'recon_x': recon_x,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def reconstruction_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss."""
        x = x.view(x.size(0), -1)  # Flatten
        return F.mse_loss(recon_x, x, reduction='mean')
    
    def kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence loss."""
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss / mu.size(0)  # Average over batch
    
    def total_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                   mu: torch.Tensor, log_var: torch.Tensor, 
                   beta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate total VAE loss."""
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = self.kl_loss(mu, log_var)
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class ConvVAE(nn.Module):
    """Convolutional VAE for image domain detection."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, 
                 image_size: int = 224):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate the size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_var = nn.Linear(self.conv_output_size, latent_dim)
        
        # Decoder
        self.decoder = self._build_decoder()
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate the size of features after encoder convolutions."""
        # This is a simplified calculation - in practice, you'd want to compute this dynamically
        size = self.image_size
        size = size // 2  # First conv
        size = size // 2  # Second conv
        size = size // 2  # Third conv
        size = size // 2  # Fourth conv
        return size * size * 128  # 128 channels in last conv layer
    
    def _build_encoder(self) -> nn.Module:
        """Build convolutional encoder."""
        return nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # 32 x 112 x 112
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # 64 x 56 x 56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # 128 x 28 x 28
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # 128 x 14 x 14
            nn.Flatten()
        )
    
    def _build_decoder(self) -> nn.Module:
        """Build convolutional decoder."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.conv_output_size),
            nn.ReLU(inplace=True),
            
            # Reshape to 128 x 14 x 14
            nn.Unflatten(1, (128, 14, 14)),
            
            # 128 x 14 x 14 -> 128 x 28 x 28
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # 128 x 28 x 28 -> 64 x 56 x 56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # 64 x 56 x 56 -> 32 x 112 x 112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # 32 x 112 x 112 -> 3 x 224 x 224
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ConvVAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        
        return {
            'recon_x': recon_x,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def reconstruction_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss."""
        return F.mse_loss(recon_x, x, reduction='mean')
    
    def kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence loss."""
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss / mu.size(0)  # Average over batch
    
    def total_loss(self, recon_x: torch.Tensor, x: torch.Tensor, 
                   mu: torch.Tensor, log_var: torch.Tensor, 
                   beta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate total VAE loss."""
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = self.kl_loss(mu, log_var)
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        } 