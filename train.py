"""
Main training script for multi-task ordinal classification.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from config import ModelConfig, TrainingConfig, DataConfig
from data_preprocessing import (
    TextPreprocessor,
    ReviewAggregator,
    PaperReviewDataset,
    load_and_preprocess_data,
    load_all_peerread_data,
    split_data
)
from model import MultiTaskOrdinalClassifier
from trainer import Trainer, create_optimizer_and_scheduler, compute_class_weights, set_seed
from metrics import compute_confusion_matrices


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

    # Create output directories
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)

    # Initialize TensorBoard logger
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

    # Load data - use combined dataset from all conferences if use_all_data is True
    if args.use_all_data:
        print("\n[*] Loading ALL PeerRead data from multiple conferences...")
        all_data = load_all_peerread_data(
            base_data_path='./data',
            text_preprocessor=text_preprocessor,
            review_aggregator=review_aggregator
        )
    else:
        print("\n[*] Loading data from single JSON file...")
        # Load and preprocess data from single file
        all_data = load_and_preprocess_data(
            data_config.data_path,
            text_preprocessor,
            review_aggregator
        )
        print(f"Loaded {len(all_data)} papers with reviews")

    # Split data
    train_data, dev_data, test_data = split_data(
        all_data,
        train_ratio=data_config.train_split,
        dev_ratio=data_config.dev_split,
        test_ratio=data_config.test_split,
        seed=training_config.seed
    )

    print(f"\n[*] Data split:")
    print(f"  Train: {len(train_data)} papers")
    print(f"  Dev: {len(dev_data)} papers")
    print(f"  Test: {len(test_data)} papers")

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)

    # Create datasets
    # train : paper + review  -> model learns what paper content earns what score
    # dev   : paper only      -> evaluation mirrors real-world inference
    # test  : paper only      -> final test mirrors real-world inference
    train_dataset = PaperReviewDataset(
        train_data,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=model_config.score_dimensions,
        inference_mode=False   # training: paper + review
    )

    dev_dataset = PaperReviewDataset(
        dev_data,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=model_config.score_dimensions,
        inference_mode=True    # eval: paper only (no review leakage)
    )

    test_dataset = PaperReviewDataset(
        test_data,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=model_config.score_dimensions,
        inference_mode=True    # test: paper only (no review leakage)
    )

    # Compute class weights if enabled
    class_weights = None
    if training_config.use_class_weights:
        print("\nComputing class weights for handling imbalance...")
        class_weights = compute_class_weights(
            train_dataset,
            model_config.score_dimensions,
            model_config.num_classes
        )
        print("Class weights computed:")
        for dim, weights in class_weights.items():
            print(f"  {dim}: {weights.numpy()}")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    print(f"\nInitializing model: {model_config.base_model_name}")
    print(f"Mode: {'Regression' if model_config.use_regression else 'Classification'}")
    model = MultiTaskOrdinalClassifier(
        base_model_name=model_config.base_model_name,
        score_dimensions=model_config.score_dimensions,
        num_classes=model_config.num_classes,
        dropout=model_config.hidden_dropout_prob,
        use_regression=model_config.use_regression
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

            for dim in model_config.score_dimensions:
                if model_config.use_regression:
                    # Continuous predictions - round for confusion matrix
                    preds = predictions[dim].cpu().numpy()
                else:
                    # Classification predictions
                    preds = torch.argmax(predictions[dim], dim=-1).cpu().numpy()
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
            all_labels_rounded[dim]      = lab_int
            all_predictions_rounded[dim] = pred_int
    else:
        all_predictions_rounded = {dim: np.round(all_predictions[dim]).astype(int)
                                   for dim in model_config.score_dimensions}
        all_labels_rounded      = {dim: np.round(all_labels[dim]).astype(int)
                                   for dim in model_config.score_dimensions}

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

