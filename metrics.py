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
    dimension_name: str = "",
    is_regression: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a single dimension.

    Args:
        y_true: True labels (continuous or discrete)
        y_pred: Predicted values (continuous for regression, discrete for classification)
        dimension_name: Name of the dimension (for logging)
        is_regression: If True, predictions are continuous and will be rounded for discrete metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if is_regression:
        # Round predictions to nearest integer for discrete metrics
        y_pred_rounded = np.round(y_pred).astype(int)
        # Round and cast true labels too (they may be float averages like 3.67)
        y_true_rounded = np.round(y_true).astype(int)
        # Clip to valid range [1, 5]
        min_val = int(np.min(y_true_rounded))
        max_val = int(np.max(y_true_rounded))
        y_pred_rounded = np.clip(y_pred_rounded, min_val, max_val)
        y_true_rounded = np.clip(y_true_rounded, 1, 5)

        # Discrete metrics use rounded integers
        metrics['accuracy'] = accuracy_score(y_true_rounded, y_pred_rounded)
        metrics['macro_f1'] = f1_score(y_true_rounded, y_pred_rounded, average='macro', zero_division=0)

        # QWK with rounded predictions
        try:
            metrics['qwk'] = quadratic_weighted_kappa(y_true_rounded, y_pred_rounded)
        except Exception:
            metrics['qwk'] = 0.0

        # Spearman correlation uses raw continuous predictions vs raw true floats
        spearman_corr, _ = spearmanr(y_true, y_pred)
        metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0

        # Regression error metrics
        metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
    else:
        # Classification mode - predictions already discrete
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            metrics['qwk'] = quadratic_weighted_kappa(y_true, y_pred)
        except:
            metrics['qwk'] = 0.0

        spearman_corr, _ = spearmanr(y_true, y_pred)
        metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))

    return metrics


def compute_multi_task_metrics(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_dimensions: List[str],
    is_regression: bool = True
) -> Dict[str, any]:
    """
    Compute metrics for all score dimensions.

    Args:
        predictions: Dict of {dimension: predicted values}
        labels: Dict of {dimension: true labels}
        score_dimensions: List of score dimension names
        is_regression: Whether predictions are continuous (regression) or discrete (classification)

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

            # Filter out missing labels (-1 or NaN)
            valid_mask = (y_true >= 0) & (~np.isnan(y_true))
            if valid_mask.sum() > 0:
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                dim_metrics = compute_metrics(y_true_valid, y_pred_valid, dim, is_regression=is_regression)
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

