"""
Configuration file for the multi-task ordinal classification pipeline.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base model — SciBERT (domain-matched, 110M params, 512-token native window)
    base_model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 2048

    # Hierarchical encoding (process long papers as 512-token chunks)
    use_hierarchical: bool = True
    use_regression: bool = False  # Classification is primary output
    use_aux_regression: bool = True  # Regression helps classification during training
    aux_regression_weight: float = 0.5  # Stronger signal -> better tie-breaker (was 0.3)
    aux_regression_loss: str = "huber"  # "huber" or "mse"
    regression_decider_enabled: bool = True  # Use regression to break close classification ties
    regression_decider_margin: float = 0.30  # Wider margin -> regression votes more often
    # If |regression - top1| >= this, trust the regression's rounded value
    # (overrides classification entirely on strong disagreements). 0 = disabled.
    regression_strong_override_distance: float = 1.5
    chunk_size: int = 512
    chunk_overlap: int = 64        # Reduced from 128 to save memory

    # Memory optimization
    gradient_checkpointing: bool = True  # Trade ~25% speed for ~30-40% less VRAM

    # Task dimensions
    score_dimensions: List[str] = None
    num_classes: int = 5  # Scores from 1 to 5 (classification outputs 0-4)

    # Model architecture
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def __post_init__(self):
        if self.score_dimensions is None:
            # Train only the primary target.
            self.score_dimensions = [
                "RECOMMENDATION",
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 2e-5
    backbone_lr: float = 5e-6  # Lower LR for pretrained backbone when unfrozen
    head_lr: float = 5e-5  # Higher LR for regression heads
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Training schedule
    num_epochs: int = 15
    train_batch_size: int = 2  # Small batch size for 8GB VRAM
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 1 * 8 = 8

    # Mixed precision for memory optimization
    fp16: bool = True  # Enable AMP (Automatic Mixed Precision)

    # Learning rate schedule
    warmup_ratio: float = 0.2  # 20% warmup for larger combined dataset
    warmup_steps: int = None  # Will be calculated from warmup_ratio if None

    # Backbone freezing
    freeze_backbone_epochs: int = 2  # Freeze encoder for first N epochs

    # Early stopping
    early_stopping_patience: int = 3   # Patience epochs before stopping
    early_stopping_metric: str = "recommendation_qwk"  # Ordinal-aware primary metric

    # Class weights (only used in classification mode)
    use_class_weights: bool = True  # Enable to handle class imbalance
    # Manual per-class weights (length = num_classes). When set, OVERRIDES the
    # inverse-frequency formula. Indices = class 0..4 (= scores 1..5).
    # Very mild bias: small nudge for class 0 and 3 so they aren't ignored,
    # but classes 1 and 2 (the bulk of the data) still own the loss surface.
    manual_class_weights: tuple = (1.5, 1.0, 1.0, 1.2, 1.0)

    # Focal loss DISABLED — it down-weights the easy majority classes (1 and 2),
    # which combined with class weights pushes the model into the extremes.
    # Re-enable later (with γ ≤ 0.5) once class weights alone are balanced.
    use_focal_loss: bool = False
    focal_gamma: float = 0.0

    # Ordinal label smoothing (Laplace-style soft labels around the true class)
    use_ordinal_smoothing: bool = True
    ordinal_smoothing: float = 0.1   # Mass moved from one-hot to neighbor distribution
    ordinal_smoothing_temperature: float = 1.0  # Sharpness of neighbor distribution

    # Weighted random sampling for minority classes — DISABLED.
    # Stacking it on top of class weights + focal loss caused the model to
    # collapse onto the boosted classes. Keep one mechanism, not three.
    use_weighted_sampler: bool = False
    sampler_minority_boost: float = 1.0

    # Logging
    logging_steps: int = 25  # More frequent logging
    eval_steps: int = 500
    save_steps: int = 1  # Save every epoch

    # Paths
    output_dir: str = "../outputs"
    log_dir: str = "../logs"

    # Seeds
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Data paths
    data_path: str = "./data/papers_reviews.json"
    train_split: float = 0.75
    dev_split: float = 0.0
    test_split: float = 0.25

    # Preprocessing
    max_paper_length: int = 10000  # characters before tokenization
    min_paper_length: int = 100
    remove_references: bool = True
    normalize_whitespace: bool = True

    # Label processing
    aggregation_method: str = "mean_round"  # mean then round to nearest int
    min_label: int = 1
    max_label: int = 5
