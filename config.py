"""
Configuration file for the multi-task ordinal classification pipeline.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base model
    base_model_name: str = "allenai/longformer-base-4096"
    max_length: int = 4096

    # For hierarchical encoding if needed
    use_hierarchical: bool = False
    use_regression: bool = True  # Use regression instead of classification
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Task dimensions
    score_dimensions: List[str] = None
    num_classes: int = 5  # Scores from 1 to 5 (used for metrics, not model output in regression mode)

    # Model architecture
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def __post_init__(self):
        if self.score_dimensions is None:
            # RECOMMENDATION is PRIMARY (index 0).
            # The rest are auxiliary — masked out for ICLR samples.
            self.score_dimensions = [
                "RECOMMENDATION",           # primary — all conferences
                "IMPACT",                   # auxiliary — ACL/CoNLL only
                "SUBSTANCE",                # auxiliary — ACL/CoNLL only
                "APPROPRIATENESS",          # auxiliary — ACL/CoNLL only
                "MEANINGFUL_COMPARISON",    # auxiliary — ACL/CoNLL only
                "SOUNDNESS_CORRECTNESS",    # auxiliary — ACL/CoNLL only
                "ORIGINALITY",              # auxiliary — ACL/CoNLL only
                "CLARITY",                  # auxiliary — ACL/CoNLL only
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
    num_epochs: int = 20
    train_batch_size: int = 1  # Small batch size for 8GB VRAM
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size = 1 * 8 = 8

    # Mixed precision for memory optimization
    fp16: bool = True  # Enable AMP (Automatic Mixed Precision)

    # Learning rate schedule
    warmup_ratio: float = 0.2  # 20% warmup for larger combined dataset
    warmup_steps: int = None  # Will be calculated from warmup_ratio if None

    # Backbone freezing
    freeze_backbone_epochs: int = 2  # Freeze encoder for first N epochs

    # Early stopping
    early_stopping_patience: int = 7  # Increased patience for better convergence
    early_stopping_metric: str = "avg_spearman"  # Average QWK across all dimensions

    # Class weights (only used in classification mode)
    use_class_weights: bool = False  # Disabled for regression

    # Logging
    logging_steps: int = 10  # More frequent logging
    eval_steps: int = 500
    save_steps: int = 1  # Save every epoch

    # Paths
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    # Seeds
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Data paths
    data_path: str = "./data/papers_reviews.json"
    train_split: float = 0.8
    dev_split: float = 0.1
    test_split: float = 0.1

    # Preprocessing
    max_paper_length: int = 10000  # characters before tokenization
    min_paper_length: int = 100
    remove_references: bool = True
    normalize_whitespace: bool = True

    # Label processing
    aggregation_method: str = "mean_round"  # mean then round to nearest int
    min_label: int = 1
    max_label: int = 5

