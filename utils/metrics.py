import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate classification metrics."""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # AUC-ROC (if probabilities are provided)
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            metrics['auc_roc'] = 0.0
    
    return metrics


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Calculate entropy of probability distributions."""
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    return entropy


def calculate_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Calculate confidence (max probability) for each sample."""
    return np.max(probabilities, axis=1)


def calculate_prototype_similarity(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between features and prototypes."""
    # Normalize features and prototypes
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    
    # Calculate cosine similarity
    similarity = np.dot(features_norm, prototypes_norm.T)
    return similarity


def calculate_domain_similarity(features: np.ndarray, domain_prototypes: np.ndarray) -> np.ndarray:
    """Calculate domain similarity scores."""
    return calculate_prototype_similarity(features, domain_prototypes)


def calculate_ukc_score(confidence: np.ndarray, similarity: np.ndarray, 
                       confidence_threshold: float = 0.8) -> np.ndarray:
    """Calculate unknown class (UKC) score."""
    # Low confidence and low similarity indicate unknown class
    confidence_score = 1.0 - confidence
    similarity_score = 1.0 - np.max(similarity, axis=1)
    
    # Combine scores
    ukc_score = 0.5 * confidence_score + 0.5 * similarity_score
    
    # Apply threshold
    ukc_score = np.where(confidence < confidence_threshold, ukc_score, 0.0)
    
    return ukc_score


def calculate_ukd_score(reconstruction_error: np.ndarray, domain_similarity: np.ndarray) -> np.ndarray:
    """Calculate unknown domain (UKD) score."""
    # High reconstruction error and low domain similarity indicate unknown domain
    reconstruction_score = reconstruction_error / (np.max(reconstruction_error) + 1e-8)
    domain_score = 1.0 - domain_similarity
    
    # Combine scores
    ukd_score = 0.5 * reconstruction_score + 0.5 * domain_score
    
    return ukd_score 