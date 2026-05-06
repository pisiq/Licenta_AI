"""
Run confusion matrices using a saved checkpoint (best_model.pt) without retraining.
"""
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import ModelConfig, DataConfig
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
from metrics import compute_confusion_matrices


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    score_dimensions = list(batch[0]["labels"].keys())

    labels = {
        dim: torch.stack([item["labels"][dim] for item in batch])
        for dim in score_dimensions
    }

    label_mask = {
        dim: torch.stack([item["label_mask"][dim] for item in batch])
        for dim in score_dimensions
    }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_mask": label_mask,
    }


def load_data(args, data_config):
    """Load data using the same split policy as training."""
    text_preprocessor = TextPreprocessor(
        normalize_whitespace=data_config.normalize_whitespace,
        remove_references=data_config.remove_references,
        max_length=data_config.max_paper_length,
        min_length=data_config.min_paper_length,
    )

    review_aggregator = ReviewAggregator(
        method=data_config.aggregation_method,
        min_val=data_config.min_label,
        max_val=data_config.max_label,
    )

    if args.use_all_data:
        all_data = load_peerread_data(
            base_data_path="data",
            text_preprocessor=text_preprocessor,
            conference_folders=PEERREAD_ALL_CONFERENCES,
            require_pdf=True,
            verbose=True,
            seed=args.seed,
        )
    else:
        all_data = load_and_preprocess_data(
            data_config.data_path,
            text_preprocessor,
            review_aggregator,
        )

    train_data, _dev_data, test_data = split_data(
        all_data,
        train_ratio=data_config.train_split,
        dev_ratio=data_config.dev_split,
        test_ratio=data_config.test_split,
        seed=args.seed,
    )

    return test_data


def main():
    parser = argparse.ArgumentParser(description="Compute confusion matrices from a saved checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to best_model.pt")
    parser.add_argument("--output_dir", type=str, default="../outputs", help="Output directory used in training")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--use_all_data", action="store_true", help="Load ALL PeerRead data")
    parser.add_argument("--base_model", type=str, default=None, help="Override base transformer model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")

    args = parser.parse_args()

    model_config = ModelConfig()
    data_config = DataConfig()

    if args.data_path:
        data_config.data_path = args.data_path
    if args.base_model:
        model_config.base_model_name = args.base_model

    checkpoint_path = args.checkpoint_path
    if not checkpoint_path:
        checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using checkpoint: {checkpoint_path}")

    test_data = load_data(args, data_config)

    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
    effective_dims = model_config.score_dimensions or SCORE_DIMENSIONS

    test_dataset = PaperReviewDataset(
        test_data,
        tokenizer,
        max_length=model_config.max_length,
        score_dimensions=effective_dims,
        inference_mode=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

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
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys:
        print(f"[WARN] Missing keys in checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[WARN] Unexpected keys in checkpoint: {load_result.unexpected_keys}")
    model.to(device)
    model.eval()

    all_predictions = {dim: [] for dim in model_config.score_dimensions}
    all_labels = {dim: [] for dim in model_config.score_dimensions}

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs["predictions"]
            regression_predictions = outputs.get("regression_predictions")

            for dim in model_config.score_dimensions:
                if model_config.use_regression:
                    preds = predictions[dim].cpu().numpy()
                else:
                    reg_preds = regression_predictions[dim] if regression_predictions is not None else None
                    pred_classes = model.resolve_class_predictions(predictions[dim], reg_preds)
                    preds = pred_classes.cpu().numpy()
                all_predictions[dim].extend(preds)
                all_labels[dim].extend(labels[dim].numpy())

    all_predictions = {dim: np.array(preds) for dim, preds in all_predictions.items()}
    all_labels = {dim: np.array(labs) for dim, labs in all_labels.items()}

    if model_config.use_regression:
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
            all_labels_rounded[dim] = lab_int - 1
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
            all_labels_rounded[dim] = lab_int - 1
            all_predictions_rounded[dim] = pred_int - 1

    confusion_matrices = compute_confusion_matrices(
        all_predictions_rounded,
        all_labels_rounded,
        model_config.score_dimensions,
        model_config.num_classes,
    )

    print("\nConfusion Matrices (rows=true, cols=predicted):")
    for dim, cm in confusion_matrices.items():
        print(f"\n{dim}:")
        print(cm)


if __name__ == "__main__":
    main()

