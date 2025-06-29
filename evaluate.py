import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config
from utils.metrics import calculate_metrics, calculate_confusion_matrix
from data.dataset import create_data_loaders
from models.ussl import USSLModel


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> USSLModel:
    """Load trained model from checkpoint."""
    model = USSLModel(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model


def evaluate_model(model: USSLModel, test_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_features = []
    all_ukc_scores = []
    all_ukd_scores = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Evaluating'):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            output = model(x)
            
            # Get predictions
            probabilities = output['probabilities']
            predictions = torch.argmax(output['logits'], dim=1)
            
            # Calculate UKC and UKD scores
            ukc_scores = model.calculate_ukc_scores(output['features'], output['logits'])
            ukd_scores = model.calculate_ukd_scores(x, output['features'])
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_features.extend(output['features'].cpu().numpy())
            all_ukc_scores.extend(ukc_scores.cpu().numpy())
            all_ukd_scores.extend(ukd_scores.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_features = np.array(all_features)
    all_ukc_scores = np.array(all_ukc_scores)
    all_ukd_scores = np.array(all_ukd_scores)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(all_labels, all_predictions)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'features': all_features,
        'ukc_scores': all_ukc_scores,
        'ukd_scores': all_ukd_scores
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ukc_ukd_distribution(ukc_scores: np.ndarray, ukd_scores: np.ndarray, 
                             labels: np.ndarray, save_path: str):
    """Plot distribution of UKC and UKD scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # UKC scores distribution
    for i in range(len(np.unique(labels))):
        mask = (labels == i)
        ax1.hist(ukc_scores[mask], alpha=0.7, label=f'Class {i}', bins=20)
    
    ax1.set_xlabel('UKC Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of UKC Scores by Class')
    ax1.legend()
    
    # UKD scores distribution
    for i in range(len(np.unique(labels))):
        mask = (labels == i)
        ax2.hist(ukd_scores[mask], alpha=0.7, label=f'Class {i}', bins=20)
    
    ax2.set_xlabel('UKD Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of UKD Scores by Class')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_visualization(features: np.ndarray, labels: np.ndarray, save_path: str):
    """Visualize features using t-SNE."""
    try:
        from sklearn.manifold import TSNE
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("sklearn not available, skipping t-SNE visualization")


def print_classification_report(labels: np.ndarray, predictions: np.ndarray, class_names: list):
    """Print detailed classification report."""
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print(report)


def save_results(results: dict, save_dir: str, config: dict):
    """Save evaluation results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write("="*30 + "\n")
        for key, value in results['metrics'].items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Save confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.txt')
    np.savetxt(cm_path, results['confusion_matrix'], fmt='%d')
    
    # Save predictions
    predictions_path = os.path.join(save_dir, 'predictions.npz')
    np.savez(predictions_path,
             predictions=results['predictions'],
             labels=results['labels'],
             probabilities=results['probabilities'],
             ukc_scores=results['ukc_scores'],
             ukd_scores=results['ukd_scores'])
    
    print(f"Results saved to {save_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate USSL Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, config, device)
    
    # Create test data loader
    # Modify config to use test data path
    config['dataset']['labeled_data_path'] = args.test_data_path
    config['dataset']['unlabeled_data_path'] = args.test_data_path
    config['dataset']['unknown_domain_path'] = args.test_data_path
    
    test_loader, _, _ = create_data_loaders(config)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in results['metrics'].items():
        print(f"{key}: {value:.4f}")
    
    # Print classification report
    if config['dataset']['name'] == 'dermatology':
        class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC', 'SCC']
    elif config['dataset']['name'] == 'ophthalmology':
        class_names = ['normal', 'amd', 'dr', 'glaucoma', 'cataract']
    else:
        class_names = [f'class_{i}' for i in range(config['dataset']['num_classes'])]
    
    print_classification_report(results['labels'], results['predictions'], class_names)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_plot_path)
    
    # Plot UKC/UKD distribution
    ukc_ukd_plot_path = os.path.join(args.save_dir, 'ukc_ukd_distribution.png')
    plot_ukc_ukd_distribution(results['ukc_scores'], results['ukd_scores'], 
                             results['labels'], ukc_ukd_plot_path)
    
    # Plot feature visualization
    feature_plot_path = os.path.join(args.save_dir, 'feature_visualization.png')
    plot_feature_visualization(results['features'], results['labels'], feature_plot_path)
    
    # Save results
    save_results(results, args.save_dir, config)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 