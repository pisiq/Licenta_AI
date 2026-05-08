"""
Configuration file for the multi-task ordinal classification pipeline.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base model — SciBERT (domain-matched, 110M params, 512-token native window)
    base_model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 2048

    # Hierarchical encoding (process long papers as 512-token chunks)
    use_hierarchical: bool = True
    chunk_aggregation: str = "attention"  # "attention" | "mean" | "max"

    # Output head type for the primary classification target.
    #   "softmax" : K-way softmax classifier
    #   "corn"    : ordinal CORN head with K-1 binary outputs (recommended for ordinal targets)
    head_type: str = "corn"

    # CORN per-head thresholds will default to 0.5 everywhere unless overridden;
    # the field below is a 5-class default kept for back-compat. When K != 5
    # the model auto-fills 0.5 for every binary head if this is the wrong length.
    # Per-binary-head thresholds for CORN at inference. Length = K-1.
    # Each head k decides whether y > k. Predicted class = 1 + #heads firing.
    #   thresholds[0] HIGHER  -> head 0 fires less   -> predict score 1 MORE
    #   thresholds[0] LOWER   -> head 0 fires more   -> predict score 1 LESS
    # To rescue class 0 (score 1), raise thresholds[0] to e.g. 0.65–0.75.
    # Defaults to 0.5 everywhere = standard CORN decoding.
    corn_thresholds: Optional[tuple] = None
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
    num_classes: int = 10  # Native ICLR scale 1..10 (classification outputs 0..9)

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
    # Strategy for computing class weights:
    #   "sqrt_inverse_freq" : weight_c ∝ 1/sqrt(count_c)   (mild, recommended)
    #   "inverse_freq"      : weight_c ∝ 1/count_c         (aggressive, can collapse)
    #   "manual"            : use `manual_class_weights` directly
    class_weight_mode: str = "sqrt_inverse_freq"
    # Optional manual override (only used when class_weight_mode == "manual").
    # If set, length must equal num_classes.
    manual_class_weights: Optional[tuple] = None
    # Extra per-class multiplier applied AFTER the chosen mode. None = uniform.
    # If set, length must equal num_classes.
    class_weight_post_boost: Optional[tuple] = None

    # Confidence-weighted training: multiply each per-reviewer training sample's
    # loss by `confidence / max_confidence` (range ~0.2..1.0). Falls back to 1.0
    # when no confidence is available. Only meaningful with expand_per_reviewer=True.
    use_confidence_weighting: bool = True

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
    max_label: int = 10
    # Target label scale: 5 (PeerRead-normalized) or 10 (native ICLR ratings).
    # Must equal model_config.num_classes for the loss/metric math to align.
    score_scale: int = 10
    # When True, restrict load_peerread_data to ICLR conferences only (drops
    # ACL/CoNLL since those don't carry confidence). When False, all configured
    # conferences are loaded.
    iclr_only: bool = True

    # If True, expand each TRAINING paper into N samples — one per reviewer's
    # individual score on the primary dimension (RECOMMENDATION). Test set is
    # never expanded; it stays at one mean-aggregated label per paper.
    expand_per_reviewer: bool = True

    # Optional per-class cap for the TRAINING set (after per-reviewer expansion).
    # If set, classes with more than this many samples are randomly subsampled
    # down to this count; rare classes (count <= cap) are kept as-is. None = off.
    # Suggested starting value: 2700 (matches the natural class-3 count).
    train_class_cap: Optional[int] = None
