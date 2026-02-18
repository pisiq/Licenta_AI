"""
Evaluation metrics for ordinal classification.
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy.stats import spearmanr


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    This is the primary metric for ordinal classification.
    Measures agreement between raters with quadratic weights for disagreement.

    Args:
        y_true: True labels (0-4 or 1-5)
        y_pred: Predicted labels (0-4 or 1-5)
        num_classes: Number of classes

    Returns:
        QWK score (0-1, higher is better)
    """
    # Use sklearn's cohen_kappa_score with quadratic weights
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dimension_name: str = ""
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a single dimension.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        dimension_name: Name of the dimension (for logging)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Macro F1
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')

    # Quadratic Weighted Kappa (primary metric)
    metrics['qwk'] = quadratic_weighted_kappa(y_true, y_pred)

    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0

    # Mean Absolute Error (for ordinal data)
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))

    return metrics


def compute_multi_task_metrics(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_dimensions: List[str]
) -> Dict[str, any]:
    """
    Compute metrics for all score dimensions.

    Args:
        predictions: Dict of {dimension: predicted labels}
        labels: Dict of {dimension: true labels}
        score_dimensions: List of score dimension names

    Returns:
        Dictionary containing:
            - per_dimension: Dict of {dimension: metrics_dict}
            - macro_avg: Dict of averaged metrics
            - avg_qwk: Average QWK across all dimensions (primary metric)
    """
    results = {
        'per_dimension': {},
        'macro_avg': {},
        'avg_qwk': 0.0
    }

    # Compute metrics for each dimension
    all_metrics = {}
    for dim in score_dimensions:
        if dim in predictions and dim in labels:
            y_true = labels[dim]
            y_pred = predictions[dim]

            # Filter out missing labels (-1)
            valid_mask = y_true >= 0
            if valid_mask.sum() > 0:
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                dim_metrics = compute_metrics(y_true_valid, y_pred_valid, dim)
                results['per_dimension'][dim] = dim_metrics

                # Collect for averaging
                for metric_name, value in dim_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)

    # Compute macro averages
    for metric_name, values in all_metrics.items():
        results['macro_avg'][metric_name] = np.mean(values)

    # Set primary metric
    results['avg_qwk'] = results['macro_avg'].get('qwk', 0.0)

    return results


def compute_confusion_matrices(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_dimensions: List[str],
    num_classes: int = 5
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrix for each dimension.

    Args:
        predictions: Dict of {dimension: predicted labels (0-4)}
        labels: Dict of {dimension: true labels (0-4)}
        score_dimensions: List of dimension names
        num_classes: Number of classes

    Returns:
        Dict of {dimension: confusion_matrix [num_classes, num_classes]}
    """
    confusion_matrices = {}

    for dim in score_dimensions:
        if dim in predictions and dim in labels:
            y_true = labels[dim]
            y_pred = predictions[dim]

            # Filter out missing labels
            valid_mask = y_true >= 0
            if valid_mask.sum() > 0:
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                cm = confusion_matrix(
                    y_true_valid,
                    y_pred_valid,
                    labels=list(range(num_classes))
                )
                confusion_matrices[dim] = cm

    return confusion_matrices


class MetricsTracker:
    """Track and format metrics during training."""

    def __init__(self, score_dimensions: List[str]):
        self.score_dimensions = score_dimensions
        self.history = {
            'train_loss': [],
            'dev_metrics': [],
            'best_avg_qwk': 0.0,
            'best_epoch': 0
        }

    def update(self, epoch: int, train_loss: float, dev_metrics: Dict):
        """Update metrics history."""
        self.history['train_loss'].append(train_loss)
        self.history['dev_metrics'].append(dev_metrics)

        # Track best performance
        avg_qwk = dev_metrics.get('avg_qwk', 0.0)
        if avg_qwk > self.history['best_avg_qwk']:
            self.history['best_avg_qwk'] = avg_qwk
            self.history['best_epoch'] = epoch

    def format_metrics(self, metrics: Dict, prefix: str = "") -> str:
        """Format metrics for logging."""
        lines = []

        if prefix:
            lines.append(f"\n{prefix}")
            lines.append("=" * 80)

        # Macro averages
        if 'macro_avg' in metrics:
            lines.append("\nMacro Averages:")
            for metric_name, value in sorted(metrics['macro_avg'].items()):
                lines.append(f"  {metric_name}: {value:.4f}")

        # Per-dimension metrics
        if 'per_dimension' in metrics:
            lines.append("\nPer-Dimension Metrics:")
            for dim in self.score_dimensions:
                if dim in metrics['per_dimension']:
                    dim_metrics = metrics['per_dimension'][dim]
                    lines.append(f"\n  {dim}:")
                    for metric_name, value in sorted(dim_metrics.items()):
                        lines.append(f"    {metric_name}: {value:.4f}")

        return "\n".join(lines)

    def get_best_metrics(self) -> Tuple[int, float]:
        """Get best epoch and QWK score."""
        return self.history['best_epoch'], self.history['best_avg_qwk']

