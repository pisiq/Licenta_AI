"""
Training script for ICLR paper review scoring with robust data loading.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from config import ModelConfig, TrainingConfig
from iclr_data_loader import ICLRDataLoader, ICLRDataset, run_sanity_checks
from model import MultiTaskOrdinalClassifier
from trainer import Trainer, create_optimizer_and_scheduler, compute_class_weights, set_seed


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    # Collect labels for each dimension
    labels = {}
    score_dimensions = batch[0]['labels'].keys()
    for dim in score_dimensions:
        labels[dim] = torch.tensor([item['labels'][dim] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def main(args):
    """Main training function with robust data loading."""

    print("\n" + "="*80)
    print("MULTI-TASK ORDINAL CLASSIFICATION FOR PAPER REVIEW SCORING")
    print("="*80 + "\n")

    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Override with command line arguments
    if args.data_path:
        data_base_path = args.data_path
    else:
        data_base_path = "C:/Facultate/Licenta/data"

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
    print(f"Device: {device}")
    print(f"Random seed: {training_config.seed}\n")

    # ============================================================================
    # PHASE 2: LOAD DATA WITH ROBUST DATA LOADER
    # ============================================================================

    print("="*80)
    print("PHASE 2: LOADING DATA")
    print("="*80)

    # Create data loader
    data_loader = ICLRDataLoader(base_path=data_base_path)

    # Load all splits
    train_papers, dev_papers, test_papers = data_loader.load_all_splits(verbose=True)

    # ============================================================================
    # PHASE 3: SANITY CHECKS
    # ============================================================================

    print("\n" + "="*80)
    print("PHASE 3: RUNNING SANITY CHECKS")
    print("="*80)

    # Run comprehensive sanity checks
    checks_passed = run_sanity_checks(train_papers, dev_papers, test_papers)

    if not checks_passed:
        print("\n❌ SANITY CHECKS FAILED! Please fix data issues before training.\n")
        return

    # ============================================================================
    # PREPARE FOR TRAINING
    # ============================================================================

    # Create output directories
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)

    # Initialize TensorBoard logger
    logger = SummaryWriter(training_config.log_dir)

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)

    # Create PyTorch datasets
    print("Creating PyTorch datasets...")
    train_dataset = ICLRDataset(
        train_papers,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=model_config.score_dimensions
    )

    dev_dataset = ICLRDataset(
        dev_papers,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=model_config.score_dimensions
    )

    test_dataset = ICLRDataset(
        test_papers,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=model_config.score_dimensions
    )

    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Dev dataset: {len(dev_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")

    # Compute class weights if enabled
    class_weights = None
    if training_config.use_class_weights:
        print("\n Computing class weights for handling imbalance...")
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
    model = MultiTaskOrdinalClassifier(
        base_model_name=model_config.base_model_name,
        score_dimensions=model_config.score_dimensions,
        num_classes=model_config.num_classes,
        dropout=model_config.hidden_dropout_prob
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

    # ============================================================================
    # TRAINING
    # ============================================================================

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    # Train
    best_model_state = trainer.train(training_config.num_epochs)

    # ============================================================================
    # EVALUATION
    # ============================================================================

    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    test_metrics = trainer.evaluate(test_dataloader)
    print(trainer.metrics_tracker.format_metrics(test_metrics, "Test Set Metrics"))

    # Compute confusion matrices
    print("\nComputing confusion matrices...")

    from metrics import compute_confusion_matrices

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
            logits = outputs['logits']

            for dim in model_config.score_dimensions:
                preds = torch.argmax(logits[dim], dim=-1).cpu().numpy()
                all_predictions[dim].extend(preds)
                all_labels[dim].extend(labels[dim].numpy())

    import numpy as np
    all_predictions = {dim: np.array(preds) for dim, preds in all_predictions.items()}
    all_labels = {dim: np.array(labs) for dim, labs in all_labels.items()}

    confusion_matrices = compute_confusion_matrices(
        all_predictions,
        all_labels,
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
            'training': training_config
        },
        'data_stats': {
            'train_size': len(train_papers),
            'dev_size': len(dev_papers),
            'test_size': len(test_papers)
        }
    }

    results_path = os.path.join(training_config.output_dir, 'test_results.pt')
    torch.save(results, results_path)
    print(f"\n✓ Test results saved to {results_path}")

    # Close logger
    logger.close()

    print("\n" + "="*80)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  - Training samples: {len(train_ids)}")
    print(f"  - Best dev QWK: {trainer.best_avg_qwk:.4f}")
    print(f"  - Test QWK: {test_metrics['avg_qwk']:.4f}")
    print(f"  - Model saved to: {training_config.output_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multi-task ordinal classifier for ICLR paper review scoring"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to data folder containing train/dev/test subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for models"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base transformer model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    main(args)

