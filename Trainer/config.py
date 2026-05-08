"""
Configuration for the CORN ordinal classification pipeline.

Pipeline summary
----------------
Encoder        : SciBERT, hierarchical with attention pooling over chunks.
Primary head   : OrdinalCORNHead (K-1 binary outputs).
Aux head       : RegressionHead (continuous Huber, used as smooth signal +
                 inference tie-breaker / strong override).
Data           : ICLR-only by default, native 1-10 ratings, per-reviewer
                 expansion on the train set, confidence-weighted loss.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbone
    base_model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 2048

    # Hierarchical encoding
    use_hierarchical: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_aggregation: str = "attention"  # "attention" | "mean" | "max"

    # Auxiliary regression head — strongly recommended ON
    use_aux_regression: bool = True
    aux_regression_weight: float = 0.5
    aux_regression_loss: str = "huber"  # "huber" | "mse"

    # Inference: regression decider
    regression_decider_enabled: bool = True
    # If |regression - top1| >= this, override CORN top1 with round(regression).
    regression_strong_override_distance: float = 1.5

    # CORN per-binary-head thresholds (length = num_classes - 1).
    # None = 0.5 everywhere. Raise thresholds[0] (e.g. 0.65) to encourage
    # predicting score 1 when the head is borderline.
    corn_thresholds: Optional[tuple] = None

    # Memory
    gradient_checkpointing: bool = True

    # Task dimensions
    score_dimensions: List[str] = None
    num_classes: int = 10  # Native ICLR scale 1..10

    # Dropout
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def __post_init__(self):
        if self.score_dimensions is None:
            self.score_dimensions = ["RECOMMENDATION"]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 2e-5
    backbone_lr: float = 5e-6
    head_lr: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 15
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Mixed precision
    fp16: bool = True

    # LR schedule
    warmup_ratio: float = 0.2
    warmup_steps: int = None

    # Backbone freeze
    freeze_backbone_epochs: int = 2

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "recommendation_qwk"

    # Confidence-weighted training: each per-reviewer training sample's loss
    # gets multiplied by `confidence/5`. Falls back to 1.0 when confidence
    # is missing. Only meaningful with `expand_per_reviewer=True`.
    use_confidence_weighting: bool = True

    # Logging
    logging_steps: int = 25
    eval_steps: int = 500
    save_steps: int = 1

    # Paths
    output_dir: str = "../outputs"
    log_dir: str = "../logs"

    # Seed
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_path: str = "./data/papers_reviews.json"
    train_split: float = 0.75
    dev_split: float = 0.0
    test_split: float = 0.25

    # Preprocessing
    max_paper_length: int = 10000
    min_paper_length: int = 100
    remove_references: bool = True
    normalize_whitespace: bool = True

    # Label processing
    aggregation_method: str = "mean_round"
    min_label: int = 1
    max_label: int = 10
    score_scale: int = 10  # must equal model_config.num_classes
    iclr_only: bool = True  # only ICLR has reviewer confidence

    # Per-reviewer expansion (TRAIN only)
    expand_per_reviewer: bool = True

    # Optional per-class cap on train set (after expansion). None = off.
    train_class_cap: Optional[int] = None
