import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer_from_scores(y_true, y_scores):
    """
    Compute Equal Error Rate (EER) from true labels and predicted scores.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_scores (array-like): Predicted scores or probabilities.

    Returns:
        eer (float): Equal Error Rate.
        eer_threshold (float): Score threshold corresponding to EER.
    """
    # Compute FPR, TPR, thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    return 1-eer, eer_threshold


def compute_eer_from_triplets(anchor_emb, positive_emb, negative_emb):
    """
    Compute EER from a batch of triplet embeddings.

    Args:
        anchor_emb (torch.Tensor): [batch_size, embedding_dim]
        positive_emb (torch.Tensor): [batch_size, embedding_dim]
        negative_emb (torch.Tensor): [batch_size, embedding_dim]

    Returns:
        eer (float): Equal Error Rate.
        eer_threshold (float): Distance threshold corresponding to EER.
    """
    y_true, y_scores = compute_dists_from_triplets(anchor_emb, positive_emb, negative_emb)
    eer, eer_threshold = compute_eer_from_scores(y_true, y_scores)
    return eer, eer_threshold


def compute_auc_from_triplets(anchor_emb, positive_emb, negative_emb):
    """
    Compute ROC AUC from a batch of triplet embeddings.

    Args:
        anchor_emb (torch.Tensor): [batch_size, embedding_dim]
        positive_emb (torch.Tensor): [batch_size, embedding_dim]
        negative_emb (torch.Tensor): [batch_size, embedding_dim]

    Returns:
        auc (float): Area Under the ROC Curve.
    """
    y_true, y_scores = compute_dists_from_triplets(anchor_emb, positive_emb, negative_emb)

    # Compute ROC AUC
    auc = roc_auc_score(y_true, y_scores)
    return auc


def compute_dists_from_triplets(anchor_emb, positive_emb, negative_emb):
    anchor_emb = anchor_emb.view(anchor_emb.size(0), -1)
    positive_emb = positive_emb.view(positive_emb.size(0), -1)
    negative_emb = negative_emb.view(negative_emb.size(0), -1)

    # Compute Euclidean distances
    pos_distances = torch.norm(anchor_emb - positive_emb, p=2, dim=1).cpu().detach().numpy()
    neg_distances = torch.norm(anchor_emb - negative_emb, p=2, dim=1).cpu().detach().numpy()

    # Labels: 1 for positive (same class), 0 for negative (different class)
    y_alike = np.concatenate([np.ones_like(pos_distances), np.zeros_like(neg_distances)])
    total_dists = np.concatenate([pos_distances, neg_distances])
    return y_alike, total_dists
