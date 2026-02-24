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
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Task dimensions
    score_dimensions: List[str] = None
    num_classes: int = 5  # Scores from 1 to 5

    # Model architecture
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def __post_init__(self):
        if self.score_dimensions is None:
            self.score_dimensions = [
                "IMPACT",
                "SUBSTANCE",
                "APPROPRIATENESS",
                "MEANINGFUL_COMPARISON",
                "SOUNDNESS_CORRECTNESS",
                "ORIGINALITY",
                "CLARITY",
                "RECOMMENDATION"
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Training schedule
    num_epochs: int = 20
    train_batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    fp16: bool = True  # Enable mixed precision
    warmup_ratio: float = 0.1

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "avg_qwk"  # Average QWK across all dimensions

    # Class weights (None = auto-compute from data)
    use_class_weights: bool = True

    # Logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500

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

