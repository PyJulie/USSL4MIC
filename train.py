import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config, parse_args, setup_experiment
from utils.metrics import calculate_metrics
from data.dataset import create_data_loaders
from models.ussl import USSLModel, USSLLoss


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Get optimizer for model."""
    if config['training']['scheduler'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['scheduler'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['scheduler']}")
    
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: dict, num_steps: int):
    """Get learning rate scheduler."""
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps
        )
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif config['training']['scheduler'] == 'warmup_cosine':
        # Warmup + cosine annealing
        def lr_lambda(step):
            if step < config['training']['warmup_epochs']:
                return step / config['training']['warmup_epochs']
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - config['training']['warmup_epochs']) / 
                                       (num_steps - config['training']['warmup_epochs'])))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model: USSLModel, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
                unknown_domain_loader: DataLoader, optimizer: optim.Optimizer, 
                loss_fn: USSLLoss, device: torch.device, epoch: int, config: dict):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_classification_loss = 0.0
    total_consistency_loss = 0.0
    total_entropy_loss = 0.0
    total_vae_loss = 0.0
    total_ukc_loss = 0.0
    total_ukd_loss = 0.0
    
    num_batches = 0
    
    # Create iterators for unlabeled and unknown domain data
    unlabeled_iter = iter(unlabeled_loader)
    unknown_domain_iter = iter(unknown_domain_loader)
    
    pbar = tqdm(labeled_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (labeled_x, labeled_y) in enumerate(pbar):
        labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
        
        # Get unlabeled data
        try:
            unlabeled_x, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_x, _ = next(unlabeled_iter)
        
        unlabeled_x = unlabeled_x.to(device)
        
        # Get unknown domain data
        try:
            unknown_x, _ = next(unknown_domain_iter)
        except StopIteration:
            unknown_domain_iter = iter(unknown_domain_loader)
            unknown_x, _ = next(unknown_domain_iter)
        
        unknown_x = unknown_x.to(device)
        
        # Forward pass for labeled data
        labeled_output = model(labeled_x)
        
        # Forward pass for unlabeled data
        unlabeled_output = model(unlabeled_x)
        
        # Forward pass for unknown domain data
        unknown_output = model(unknown_x)
        
        # Calculate UKC and UKD scores
        ukc_scores = model.calculate_ukc_scores(
            unlabeled_output['features'], unlabeled_output['logits']
        )
        ukd_scores = model.calculate_ukd_scores(
            unknown_x, unknown_output['features']
        )
        
        # Create consistency targets (weak augmentation)
        # For simplicity, we'll use the same images with different augmentations
        # In practice, you'd want to apply different augmentation strategies
        consistency_output = model(unlabeled_x)  # This should be weak augmentation
        
        # Calculate losses
        loss, losses = loss_fn.total_loss(
            labeled_output, labeled_y, labeled_x,
            ukc_scores, ukd_scores, consistency_output['logits']
        )
        
        # Add consistency loss for unlabeled data
        consistency_loss = loss_fn.consistency_loss(
            unlabeled_output['logits'], consistency_output['logits']
        )
        loss += config['training']['ssl']['lambda_consistency'] * consistency_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update prototypes
        model.update_prototypes(labeled_output['features'], labeled_y)
        
        # Update statistics
        total_loss += loss.item()
        total_classification_loss += losses.get('classification', 0.0)
        total_consistency_loss += consistency_loss.item()
        total_entropy_loss += losses.get('entropy', 0.0)
        total_vae_loss += losses.get('vae_reconstruction', 0.0) + losses.get('vae_kl', 0.0)
        total_ukc_loss += losses.get('ukc_weighted', 0.0)
        total_ukd_loss += losses.get('ukd_weighted', 0.0)
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Cls': f'{losses.get("classification", 0.0):.4f}',
            'Cons': f'{consistency_loss.item():.4f}',
            'Ent': f'{losses.get("entropy", 0.0):.4f}'
        })
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches
    avg_consistency_loss = total_consistency_loss / num_batches
    avg_entropy_loss = total_entropy_loss / num_batches
    avg_vae_loss = total_vae_loss / num_batches
    avg_ukc_loss = total_ukc_loss / num_batches
    avg_ukd_loss = total_ukd_loss / num_batches
    
    return {
        'loss': avg_loss,
        'classification_loss': avg_classification_loss,
        'consistency_loss': avg_consistency_loss,
        'entropy_loss': avg_entropy_loss,
        'vae_loss': avg_vae_loss,
        'ukc_loss': avg_ukc_loss,
        'ukd_loss': avg_ukd_loss
    }


def validate(model: USSLModel, val_loader: DataLoader, device: torch.device) -> dict:
    """Validate the model."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='Validation'):
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            probabilities = output['probabilities']
            predictions = torch.argmax(output['logits'], dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    return metrics


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   metrics: dict, config: dict, save_path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: str):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}")
    
    return epoch, metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = setup_experiment(config, args)
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Set device
    device = torch.device(f"cuda:{config['experiment']['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    labeled_loader, unlabeled_loader, unknown_domain_loader = create_data_loaders(config)
    
    # Create model
    print("Creating model...")
    model = USSLModel(config).to(device)
    
    # Create loss function
    loss_fn = USSLLoss(config)
    
    # Create optimizer
    optimizer = get_optimizer(model, config)
    
    # Create scheduler
    num_steps = len(labeled_loader) * config['training']['epochs']
    scheduler = get_scheduler(optimizer, config, num_steps)
    
    # Initialize logging
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])
    else:
        writer = None
    
    if config['logging']['wandb']:
        wandb.init(project="ussl", config=config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume:
        start_epoch, best_metrics = load_checkpoint(model, optimizer, args.resume)
        best_metric = best_metrics.get('accuracy', 0.0)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train for one epoch
        train_metrics = train_epoch(
            model, labeled_loader, unlabeled_loader, unknown_domain_loader,
            optimizer, loss_fn, device, epoch, config
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Log training metrics
        if writer:
            for key, value in train_metrics.items():
                writer.add_scalar(f'train/{key}', value, epoch)
        
        if config['logging']['wandb']:
            wandb.log({f'train/{key}': value for key, value in train_metrics.items()}, step=epoch)
        
        # Validation
        if epoch % config['logging']['eval_freq'] == 0:
            # For now, we'll use labeled data as validation
            # In practice, you'd want a separate validation set
            val_metrics = validate(model, labeled_loader, device)
            
            # Log validation metrics
            if writer:
                for key, value in val_metrics.items():
                    writer.add_scalar(f'val/{key}', value, epoch)
            
            if config['logging']['wandb']:
                wandb.log({f'val/{key}': value for key, value in val_metrics.items()}, step=epoch)
            
            print(f"Epoch {epoch}: Val Accuracy = {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_metric:
                best_metric = val_metrics['accuracy']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, config,
                    os.path.join(config['checkpoint']['save_dir'], 'best_model.pth')
                )
        
        # Save checkpoint periodically
        if epoch % config['logging']['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics, config,
                os.path.join(config['checkpoint']['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, config['training']['epochs'] - 1, train_metrics, config,
        os.path.join(config['checkpoint']['save_dir'], 'final_model.pth')
    )
    
    print("Training completed!")
    
    if writer:
        writer.close()
    
    if config['logging']['wandb']:
        wandb.finish()


if __name__ == "__main__":
    main() 