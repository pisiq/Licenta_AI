"""
Main training script for multi-task ordinal classification.
"""
import os
import sys
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import json

from config import ModelConfig, TrainingConfig, DataConfig
from data_preprocessing import (
    TextPreprocessor,
    ReviewAggregator,
    PaperReviewDataset,
    load_peerread_data,
    load_and_preprocess_data,
    split_data,
    SCORE_DIMENSIONS,
    PEERREAD_ALL_CONFERENCES,
)
from model import MultiTaskOrdinalClassifier
from trainer import (
    Trainer,
    create_optimizer_and_scheduler,
    compute_class_weights,
    make_weighted_sampler,
    set_seed,
)
from metrics import compute_confusion_matrices


class _Tee:
    """Duplicate stdout/stderr writes into a file handle for persistent text logs."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _setup_run_dirs(training_config) -> tuple:
    """Create timestamped run directories and tee stdout/stderr to a log file.

    Resolves relative output_dir / log_dir against the *project root* (parent
    of the Trainer/ directory) so logs always land in the project regardless
    of where the script is invoked from.

    Returns (run_output_dir, run_log_dir, log_file_handle).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _abspath(p: str) -> str:
        if os.path.isabs(p):
            return os.path.normpath(p)
        # Strip any leading "../" — they were placeholders from the old CWD-relative defaults.
        cleaned = p.replace("\\", "/").lstrip("./")
        while cleaned.startswith("../"):
            cleaned = cleaned[3:]
        return os.path.normpath(os.path.join(project_root, cleaned))

    base_output_dir = _abspath(training_config.output_dir)
    base_log_dir    = _abspath(training_config.log_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    run_log_dir    = os.path.join(base_log_dir,    f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    log_file_path = os.path.join(run_log_dir, "train.log")
    log_fh = open(log_file_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    print(f"[OK] Run output dir : {run_output_dir}")
    print(f"[OK] Run log dir    : {run_log_dir}")
    print(f"[OK] Text log file  : {log_file_path}")

    # Repoint training_config so checkpoints land in the per-run folder
    training_config.output_dir = run_output_dir
    training_config.log_dir = run_log_dir
    return run_output_dir, run_log_dir, log_fh


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids      = torch.stack([item['input_ids']      for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    score_dimensions = list(batch[0]['labels'].keys())

    # Labels: stack pre-built float tensors (NaN where score is missing)
    labels = {
        dim: torch.stack([item['labels'][dim] for item in batch])
        for dim in score_dimensions
    }

    # Mask: stack pre-built float tensors (1.0 valid, 0.0 missing)
    label_mask = {
        dim: torch.stack([item['label_mask'][dim] for item in batch])
        for dim in score_dimensions
    }

    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
        'label_mask':     label_mask,
    }


def main(args):
    """Main training function."""

    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    # Override with command line arguments if provided
    if args.data_path:
        data_config.data_path = args.data_path
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.base_model:
        model_config.base_model_name = args.base_model
    if args.batch_size:
        training_config.train_batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.num_epochs:
        training_config.num_epochs = args.num_epochs

    # Set seed for reproducibility
    set_seed(training_config.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Create per-run output/log directories and start text log
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    _setup_run_dirs(training_config)

    # Initialize TensorBoard logger (writes into per-run log dir)
    logger = SummaryWriter(training_config.log_dir)

    print("Loading and preprocessing data...")

    # Create preprocessors
    text_preprocessor = TextPreprocessor(
        normalize_whitespace=data_config.normalize_whitespace,
        remove_references=data_config.remove_references,
        max_length=data_config.max_paper_length,
        min_length=data_config.min_paper_length
    )

    review_aggregator = ReviewAggregator(
        method=data_config.aggregation_method,
        min_val=data_config.min_label,
        max_val=data_config.max_label
    )

    # Load data
    if args.use_all_data:
        print("\n[*] Loading ALL PeerRead data (ACL 2017, CoNLL 2016, ICLR 2017-2020)...")
        all_data = load_peerread_data(
            base_data_path    ='data',
            text_preprocessor = text_preprocessor,
            conference_folders = PEERREAD_ALL_CONFERENCES,
            require_pdf       = True,
            verbose           = True,
            seed              = training_config.seed,
        )
    else:
        print("\n[*] Loading data from single JSON file...")
        all_data = load_and_preprocess_data(
            data_config.data_path,
            text_preprocessor,
            review_aggregator
        )
        print(f"Loaded {len(all_data)} papers with reviews")

    # Split data
    train_data, _dev_data, test_data = split_data(
        all_data,
        train_ratio=data_config.train_split,
        dev_ratio=data_config.dev_split,
        test_ratio=data_config.test_split,
        seed=training_config.seed
    )

    print(f"\n[*] Data split:")
    print(f"  Train: {len(train_data)} papers")
    print(f"  Test: {len(test_data)} papers")

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)

    # Create datasets
    # train : paper + review  -> model learns what paper content earns what score
    # test  : paper only      -> final test mirrors real-world inference
    effective_dims = model_config.score_dimensions or SCORE_DIMENSIONS
    train_dataset = PaperReviewDataset(
        train_data,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=effective_dims,
        inference_mode=False   # training: paper + review
    )

    test_dataset = PaperReviewDataset(
        test_data,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=effective_dims,
        inference_mode=True    # test: paper only (no review leakage)
    )

    # Compute class weights if enabled
    class_weights = None
    if training_config.use_class_weights:
        manual = getattr(training_config, 'manual_class_weights', None)
        if manual is not None:
            manual_t = torch.tensor(list(manual), dtype=torch.float32)
            assert manual_t.numel() == model_config.num_classes, (
                f"manual_class_weights length ({manual_t.numel()}) must equal "
                f"num_classes ({model_config.num_classes})"
            )
            class_weights = {dim: manual_t.clone() for dim in model_config.score_dimensions}
            print("\nUsing MANUAL class weights:")
            for dim, weights in class_weights.items():
                print(f"  {dim}: {weights.numpy()}")
        else:
            print("\nComputing class weights from inverse frequency...")
            class_weights = compute_class_weights(
                train_dataset,
                model_config.score_dimensions,
                model_config.num_classes
            )
            print("Class weights computed:")
            for dim, weights in class_weights.items():
                print(f"  {dim}: {weights.numpy()}")

    # Create data loaders
    # num_workers=0 on Windows (avoids multiprocessing spawn issues)
    # pin_memory=True speeds up CPU→GPU transfers on CUDA
    _pin = device.type == 'cuda'

    train_sampler = None
    if getattr(training_config, 'use_weighted_sampler', False):
        print("\nBuilding WeightedRandomSampler for class-balanced batches...")
        train_sampler = make_weighted_sampler(
            train_dataset,
            primary_dim="RECOMMENDATION",
            num_classes=model_config.num_classes,
            minority_boost=getattr(training_config, 'sampler_minority_boost', 1.0),
            minority_classes=(0, 3),
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=_pin,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=_pin,
    )

    # Use test split as dev for early stopping/best-model saving
    dev_dataloader = test_dataloader

    # Create model
    print(f"\nInitializing model: {model_config.base_model_name}")
    print(f"Mode: {'Regression' if model_config.use_regression else 'Classification'}")
    model = MultiTaskOrdinalClassifier(
        base_model_name=model_config.base_model_name,
        score_dimensions=model_config.score_dimensions,
        num_classes=model_config.num_classes,
        dropout=model_config.hidden_dropout_prob,
        use_regression=model_config.use_regression,
        use_aux_regression=model_config.use_aux_regression,
        aux_regression_weight=model_config.aux_regression_weight,
        aux_regression_loss=model_config.aux_regression_loss,
        use_hierarchical=model_config.use_hierarchical,
        chunk_size=model_config.chunk_size,
        regression_decider_enabled=model_config.regression_decider_enabled,
        regression_decider_margin=model_config.regression_decider_margin,
        regression_strong_override_distance=getattr(model_config, 'regression_strong_override_distance', 0.0),
        use_focal_loss=getattr(training_config, 'use_focal_loss', False),
        focal_gamma=getattr(training_config, 'focal_gamma', 2.0),
        use_ordinal_smoothing=getattr(training_config, 'use_ordinal_smoothing', False),
        ordinal_smoothing=getattr(training_config, 'ordinal_smoothing', 0.1),
        ordinal_smoothing_temperature=getattr(training_config, 'ordinal_smoothing_temperature', 1.0),
        gradient_checkpointing=getattr(model_config, 'gradient_checkpointing', False),
    )

    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        train_dataloader,
        training_config
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=training_config,
        class_weights=class_weights,
        logger=logger
    )

    # Train
    best_model_state = trainer.train(training_config.num_epochs)

    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)

    test_metrics = trainer.evaluate(test_dataloader)
    print(trainer.metrics_tracker.format_metrics(test_metrics, "Test Set Metrics"))

    if logger:
        step = training_config.num_epochs
        logger.add_scalar('test/recommendation_spearman', test_metrics.get('recommendation_spearman', 0.0), step)
        logger.add_scalar('test/recommendation_qwk',      test_metrics.get('recommendation_qwk', 0.0),      step)
        logger.add_scalar('test/recommendation_mae',       test_metrics.get('recommendation_mae', 5.0),       step)
        logger.add_scalar('test/recommendation_accuracy',  test_metrics.get('recommendation_accuracy', 0.0),  step)
        logger.add_scalar('test/recommendation_macro_f1',  test_metrics.get('recommendation_macro_f1', 0.0),  step)
        logger.add_scalar('test/recommendation_rmse',      test_metrics.get('recommendation_rmse', 0.0),      step)
        logger.add_scalar('test/avg_accuracy', test_metrics.get('macro_avg', {}).get('accuracy', 0.0), step)

    # Compute confusion matrices (for rounded predictions in regression mode)
    print("\nComputing confusion matrices...")

    # Get predictions for test set
    model.eval()
    all_predictions = {dim: [] for dim in model_config.score_dimensions}
    all_labels = {dim: [] for dim in model_config.score_dimensions}

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs['predictions']
            regression_predictions = outputs.get('regression_predictions')

            for dim in model_config.score_dimensions:
                if model_config.use_regression:
                    # Continuous predictions - round for confusion matrix
                    preds = predictions[dim].cpu().numpy()
                else:
                    # Classification predictions (regression-guided if enabled)
                    reg_preds = regression_predictions[dim] if regression_predictions is not None else None
                    pred_classes = model.resolve_class_predictions(predictions[dim], reg_preds)
                    preds = pred_classes.cpu().numpy()
                all_predictions[dim].extend(preds)
                all_labels[dim].extend(labels[dim].numpy())

    import numpy as np
    all_predictions = {dim: np.array(preds) for dim, preds in all_predictions.items()}
    all_labels      = {dim: np.array(labs)  for dim, labs  in all_labels.items()}

    # Filter NaN labels and round both labels and preds to int for confusion matrix
    if model_config.use_regression:
        all_predictions_rounded = {}
        all_labels_rounded = {}
        for dim in model_config.score_dimensions:
            lab = all_labels[dim]
            pred = all_predictions[dim]
            valid_mask = (~np.isnan(lab)) & (lab >= 1)
            if valid_mask.sum() > 0:
                lab_int  = np.clip(np.round(lab[valid_mask]).astype(int),  1, 5)
                pred_int = np.clip(np.round(pred[valid_mask]).astype(int), 1, 5)
            else:
                lab_int, pred_int = np.array([], dtype=int), np.array([], dtype=int)
            all_labels_rounded[dim]      = lab_int - 1
            all_predictions_rounded[dim] = pred_int - 1
    else:
        all_predictions_rounded = {}
        all_labels_rounded = {}
        for dim in model_config.score_dimensions:
            lab = all_labels[dim]
            pred = all_predictions[dim]
            valid_mask = (~np.isnan(lab)) & (lab >= 1)
            if valid_mask.sum() > 0:
                lab_int = np.clip(np.round(lab[valid_mask]).astype(int), 1, 5)
                pred_int = np.clip(np.round(pred[valid_mask]).astype(int), 1, 5)
            else:
                lab_int, pred_int = np.array([], dtype=int), np.array([], dtype=int)
            all_labels_rounded[dim]      = lab_int - 1
            all_predictions_rounded[dim] = pred_int - 1

    confusion_matrices = compute_confusion_matrices(
        all_predictions_rounded,
        all_labels_rounded,
        model_config.score_dimensions,
        model_config.num_classes
    )

    print("\nConfusion Matrices (rows=true, cols=predicted):")
    for dim, cm in confusion_matrices.items():
        print(f"\n{dim}:")
        print(cm)

    # Save final results
    results = {
        'test_metrics': test_metrics,
        'confusion_matrices': confusion_matrices,
        'config': {
            'model': model_config,
            'training': training_config,
            'data': data_config
        }
    }

    results_path = os.path.join(training_config.output_dir, 'test_results.pt')
    torch.save(results, results_path)
    print(f"\n[OK] Test results saved to {results_path}")

    # Save JSON-friendly results for easy inspection
    json_results = {
        'test_metrics': test_metrics,
        'confusion_matrices': {
            k: v.tolist() for k, v in confusion_matrices.items()
        },
    }
    json_path = os.path.join(training_config.output_dir, 'test_results.json')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"[OK] Test results JSON saved to {json_path}")

    # Close logger
    logger.close()

    print("\n" + "="*80)
    print("Training pipeline completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task ordinal classifier for paper review scoring")

    parser.add_argument("--data_path", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--use_all_data", action="store_true", help="Load ALL PeerRead data from all conference folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for models")
    parser.add_argument("--base_model", type=str, default=None, help="Base transformer model")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")

    args = parser.parse_args()

    main(args)
