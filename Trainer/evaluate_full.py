"""
Full-dataset evaluation for a saved CORN model.

Loads `best_model.pt`, runs the forward pass on the FULL dataset (train +
test, no per-reviewer expansion — one prediction per paper using the mean
reviewer rating as the label), and reports:

  - Exact accuracy
  - ±1 accuracy   (fraction of predictions within 1 of truth)
  - ±2 accuracy   (fraction within 2 of truth)
  - QWK, Spearman, MAE   (ordinal context)

Reports broken down by predefined split (train / test / combined) so you
can see fit vs generalization at a glance.

This script is read-only — it does not modify config.py or any training state.

Usage
-----
  python Trainer/evaluate_full.py --model_path outputs/.../best_model.pt
  python Trainer/evaluate_full.py --model_path .../best_model.pt --json
"""
import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config import ModelConfig, TrainingConfig, DataConfig
from data_preprocessing import (
    TextPreprocessor,
    PaperReviewDataset,
    load_peerread_data,
    PEERREAD_ALL_CONFERENCES,
)
from model import MultiTaskOrdinalClassifier
from metrics import quadratic_weighted_kappa
from scipy.stats import spearmanr
from train import collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(mc: ModelConfig):
    return MultiTaskOrdinalClassifier(
        base_model_name=mc.base_model_name,
        score_dimensions=mc.score_dimensions,
        num_classes=mc.num_classes,
        dropout=mc.hidden_dropout_prob,
        use_aux_regression=mc.use_aux_regression,
        aux_regression_weight=mc.aux_regression_weight,
        aux_regression_loss=mc.aux_regression_loss,
        use_hierarchical=mc.use_hierarchical,
        chunk_size=mc.chunk_size,
        chunk_aggregation=mc.chunk_aggregation,
        regression_decider_enabled=mc.regression_decider_enabled,
        regression_strong_override_distance=mc.regression_strong_override_distance,
        corn_thresholds=(
            list(mc.corn_thresholds) if mc.corn_thresholds is not None else None
        ),
    )


def _accuracy_within(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Fraction of predictions within `k` of the truth (inclusive)."""
    if len(y_true) == 0:
        return 0.0
    diff = np.abs(y_true.astype(int) - y_pred.astype(int))
    return float((diff <= k).mean())


def _evaluate_split(model, papers, tokenizer, mc, tc, device, split_name):
    """Build a dataset over `papers`, run the model, return metrics + arrays."""
    if not papers:
        return None
    dataset = PaperReviewDataset(
        papers, tokenizer,
        max_length=mc.max_length,
        score_dimensions=mc.score_dimensions,
        inference_mode=True,         # paper only, no review leakage
        print_summary=False,
    )
    pin = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=tc.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin,
    )

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split_name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['predictions']['RECOMMENDATION']
            reg = out.get('regression_predictions', {}).get('RECOMMENDATION')
            cls = model.resolve_class_predictions(logits, reg).cpu().numpy()
            all_preds.append(cls)
            all_labels.append(batch['labels']['RECOMMENDATION'].numpy())

    pred = np.concatenate(all_preds)                    # int [N]
    lab  = np.concatenate(all_labels)                   # float [N], possibly NaN

    valid = (~np.isnan(lab)) & (lab >= 1)
    pred = pred[valid]
    lab_int = np.clip(np.round(lab[valid]).astype(int), 1, mc.num_classes)
    lab_raw = lab[valid]   # keep continuous floats for Spearman

    if len(pred) == 0:
        return None

    metrics = {
        "n":            int(len(pred)),
        "accuracy":     float((pred == lab_int).mean()),
        "acc_within_1": _accuracy_within(lab_int, pred, 1),
        "acc_within_2": _accuracy_within(lab_int, pred, 2),
        "mae":          float(np.mean(np.abs(pred.astype(float) - lab_raw))),
        "qwk":          float(quadratic_weighted_kappa(lab_int, pred, num_classes=mc.num_classes)),
        "spearman":     float(spearmanr(lab_raw, pred).correlation or 0.0),
    }
    return metrics, pred, lab_int


def _format(label: str, m: dict) -> str:
    return (f"  {label:<14}  n={m['n']:>5}  "
            f"acc={m['accuracy']:.4f}  "
            f"±1={m['acc_within_1']:.4f}  "
            f"±2={m['acc_within_2']:.4f}  "
            f"qwk={m['qwk']:.4f}  "
            f"mae={m['mae']:.4f}  "
            f"spearman={m['spearman']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to best_model.pt")
    parser.add_argument("--json", action="store_true", help="Print results as JSON instead.")
    args = parser.parse_args()

    mc = ModelConfig(); tc = TrainingConfig(); dc = DataConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load full dataset (NO per-reviewer expansion — we want one sample per
    # paper using the mean rating, mirroring inference at deployment time)
    text_prep = TextPreprocessor(
        normalize_whitespace=dc.normalize_whitespace,
        remove_references=dc.remove_references,
        max_length=dc.max_paper_length,
        min_length=dc.min_paper_length,
    )
    print("\nLoading full dataset...")
    all_data = load_peerread_data(
        base_data_path='data',
        text_preprocessor=text_prep,
        conference_folders=PEERREAD_ALL_CONFERENCES,
        require_pdf=True,
        verbose=True,
        seed=tc.seed,
        target_scale=float(dc.score_scale),
        iclr_only=dc.iclr_only,
    )
    train_papers = [p for p in all_data if p.split == "train"]
    test_papers  = [p for p in all_data if p.split in ("dev", "test")]
    print(f"\nFound {len(train_papers)} train and {len(test_papers)} test papers (total {len(all_data)})")

    # Build + load model
    print(f"\nLoading model: {args.model_path}")
    model = _build_model(mc)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f"[WARN] missing keys: {len(res.missing_keys)}")
    if res.unexpected_keys:
        print(f"[WARN] unexpected keys: {len(res.unexpected_keys)}")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(mc.base_model_name)

    print("\nEvaluating per split...")
    train_out = _evaluate_split(model, train_papers, tokenizer, mc, tc, device, "train")
    test_out  = _evaluate_split(model, test_papers,  tokenizer, mc, tc, device, "test")

    splits = {}
    combined_pred = []
    combined_lab  = []

    if train_out is not None:
        splits["train"] = train_out[0]
        combined_pred.append(train_out[1])
        combined_lab.append(train_out[2])
    if test_out is not None:
        splits["test"]  = test_out[0]
        combined_pred.append(test_out[1])
        combined_lab.append(test_out[2])

    if combined_pred:
        cp = np.concatenate(combined_pred)
        cl = np.concatenate(combined_lab)
        try:
            sp = float(spearmanr(cl, cp).correlation or 0.0)
        except Exception:
            sp = 0.0
        splits["all"] = {
            "n":            int(len(cp)),
            "accuracy":     float((cp == cl).mean()),
            "acc_within_1": _accuracy_within(cl, cp, 1),
            "acc_within_2": _accuracy_within(cl, cp, 2),
            "mae":          float(np.mean(np.abs(cp.astype(float) - cl.astype(float)))),
            "qwk":          float(quadratic_weighted_kappa(cl, cp, num_classes=mc.num_classes)),
            "spearman":     sp,
        }

    if args.json:
        print(json.dumps(splits, indent=2))
        return

    print("\n" + "=" * 110)
    print(f"FULL-DATASET EVAL — {os.path.basename(args.model_path)}")
    print("=" * 110)
    for name in ("train", "test", "all"):
        if name in splits:
            print(_format(name, splits[name]))
    print("=" * 110)

    # Per-class confusion (combined)
    if "all" in splits and combined_pred:
        cp = np.concatenate(combined_pred)
        cl = np.concatenate(combined_lab)
        K = mc.num_classes
        print("\nConfusion matrix on FULL dataset (rows=true 1..K, cols=pred 1..K):")
        cm = np.zeros((K, K), dtype=int)
        for t, p in zip(cl, cp):
            cm[int(t) - 1, int(p) - 1] += 1
        for row in cm:
            print("  " + " ".join(f"{int(v):5d}" for v in row))

        # Per-class ±1 hit rate (rare-class diagnostic)
        print("\nPer-class ±1 accuracy:")
        for c in range(1, K + 1):
            mask = cl == c
            if mask.sum() == 0:
                print(f"  score {c:>2}:  n=0")
                continue
            within1 = _accuracy_within(cl[mask], cp[mask], 1)
            print(f"  score {c:>2}:  n={int(mask.sum()):>5}  ±1 acc = {within1:.4f}")


if __name__ == "__main__":
    main()
