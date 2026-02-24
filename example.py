"""
Example script demonstrating the complete pipeline.

This script shows how to:
1. Generate sample data
2. Train a model
3. Make predictions
"""
import os
import torch
from transformers import AutoTokenizer

from config import ModelConfig, TrainingConfig, DataConfig
from data_preprocessing import (
    TextPreprocessor,
    ReviewAggregator,
    PaperReviewDataset,
    load_and_preprocess_data,
    split_data
)
from model import MultiTaskOrdinalClassifier
from trainer import Trainer, create_optimizer_and_scheduler, compute_class_weights, set_seed
from inference import PaperReviewPredictor, load_model_for_inference
from torch.utils.data import DataLoader


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    labels = {}
    score_dimensions = batch[0]['labels'].keys()
    for dim in score_dimensions:
        labels[dim] = torch.tensor([item['labels'][dim] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def main():
    """Run complete example pipeline."""

    print("\n" + "="*80)
    print("MULTI-TASK ORDINAL CLASSIFICATION PIPELINE - EXAMPLE")
    print("="*80 + "\n")



    # ========== STEP 2: Setup Configuration ==========
    print("\nSTEP 2: Setting up configuration...")

    # Use smaller model for faster demo
    model_config = ModelConfig()
    model_config.base_model_name = "distilbert-base-uncased"  # Smaller/faster for demo
    model_config.max_length = 512

    training_config = TrainingConfig()
    training_config.num_epochs = 3  # Fewer epochs for demo
    training_config.train_batch_size = 4
    training_config.eval_batch_size = 8
    training_config.output_dir = "./demo_outputs"
    training_config.log_dir = "./demo_logs"

    data_config = DataConfig()
    # data_config.data_path defaults to "./data/papers_reviews.json"

    set_seed(training_config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========== STEP 3: Load and Preprocess Data ==========
    print("\nSTEP 3: Loading and preprocessing data...")

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

    all_data = load_and_preprocess_data(
        data_config.data_path,
        text_preprocessor,
        review_aggregator
    )

    print(f"Loaded {len(all_data)} papers")

    train_data, dev_data, test_data = split_data(
        all_data,
        train_ratio=data_config.train_split,
        dev_ratio=data_config.dev_split,
        test_ratio=data_config.test_split,
        seed=training_config.seed
    )

    print(f"Split: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")

    # ========== STEP 4: Create Datasets ==========
    print("\nSTEP 4: Creating datasets...")

    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)

    train_dataset = PaperReviewDataset(
        train_data, tokenizer, model_config.max_length, model_config.score_dimensions
    )
    dev_dataset = PaperReviewDataset(
        dev_data, tokenizer, model_config.max_length, model_config.score_dimensions
    )
    test_dataset = PaperReviewDataset(
        test_data, tokenizer, model_config.max_length, model_config.score_dimensions
    )

    # ========== STEP 5: Create Model ==========
    print("\nSTEP 5: Creating model...")

    model = MultiTaskOrdinalClassifier(
        base_model_name=model_config.base_model_name,
        score_dimensions=model_config.score_dimensions,
        num_classes=model_config.num_classes,
        dropout=model_config.hidden_dropout_prob
    )

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== STEP 6: Setup Training ==========
    print("\nSTEP 6: Setting up training...")

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

    # Compute class weights
    class_weights = None
    if training_config.use_class_weights:
        class_weights = compute_class_weights(
            train_dataset,
            model_config.score_dimensions,
            model_config.num_classes
        )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, train_dataloader, training_config
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
        logger=None  # No TensorBoard for demo
    )

    # ========== STEP 7: Train Model ==========
    print("\nSTEP 7: Training model...")
    print("(This may take several minutes depending on your hardware)\n")

    best_model_state = trainer.train(training_config.num_epochs)

    # ========== STEP 8: Evaluate on Test Set ==========
    print("\nSTEP 8: Evaluating on test set...")

    test_metrics = trainer.evaluate(test_dataloader)
    print(trainer.metrics_tracker.format_metrics(test_metrics, "Test Set Results"))

    # ========== STEP 9: Make Predictions ==========
    print("\nSTEP 9: Making predictions on a sample paper...")

    # Create predictor
    predictor = PaperReviewPredictor(
        model=model,
        tokenizer=tokenizer,
        text_preprocessor=text_preprocessor,
        device=device
    )

    # Get a sample paper from test set
    sample_paper = test_data[0]

    print(f"\nSample Paper Title: {sample_paper.title[:100]}...")
    print(f"\nTrue Scores:")
    for dim, score in sample_paper.scores.items():
        print(f"  {dim}: {score}")

    # Predict
    predictions = predictor.predict_scores(
        paper_text=sample_paper.full_text,
        title=sample_paper.title,
        abstract=sample_paper.abstract
    )

    print(f"\nPredicted Scores:")
    for dim, score in predictions.items():
        print(f"  {dim}: {score}")

    # Get probabilities
    probs = predictor.predict_probabilities(
        paper_text=sample_paper.full_text,
        title=sample_paper.title,
        abstract=sample_paper.abstract
    )

    print(f"\nPrediction Probabilities (first dimension as example):")
    first_dim = list(probs.keys())[0]
    print(f"  {first_dim}:")
    for score, prob in probs[first_dim].items():
        print(f"    {score}: {prob:.3f}")

    # ========== STEP 10: Save Model ==========
    print(f"\nSTEP 10: Saving model...")
    os.makedirs(training_config.output_dir, exist_ok=True)
    model_path = os.path.join(training_config.output_dir, "demo_model.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'metrics': test_metrics
    }, model_path)

    print(f"Model saved to: {model_path}")

    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  - Trained on {len(train_data)} papers")
    print(f"  - Best dev QWK: {trainer.best_avg_qwk:.4f}")
    print(f"  - Test QWK: {test_metrics['avg_qwk']:.4f}")
    print(f"  - Model saved to: {model_path}")
    print("\nYou can now use this model for inference on new papers!")


if __name__ == "__main__":
    main()

