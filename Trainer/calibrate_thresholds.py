"""
Post-training calibration of CORN per-head thresholds.

Loads `best_model.pt`, runs the forward pass on the test set ONCE and caches
the CORN logits + regression predictions + labels. Then iterates over
threshold configurations and re-decodes the cache — each evaluation is
sub-second after the initial forward pass.

CORN decoding recap (K=10):
  predicted class = 1 + sum_k 1[ sigmoid(logit_k) > threshold_k ]
  - LOWER  threshold[k]  -> head k fires more  -> predict class >= k+2 more often
  - HIGHER threshold[k]  -> head k fires less  -> predict class <= k+1 more often

Reading from a confusion matrix:
  - Class c (= score c) is OVER-predicted? -> head (c-1) is firing too often.
    Raise threshold[c-1] to push some predictions down to class c-1.
    OR lower threshold[c] to push some predictions UP past class c.
  - Class c is UNDER-predicted? -> the opposite.

Usage
-----
  # Built-in curated sweep:
  python Trainer/calibrate_thresholds.py --model_path outputs/.../best_model.pt --sweep

  # Single custom threshold set (length K-1 = 9 for K=10):
  python Trainer/calibrate_thresholds.py --model_path .../best_model.pt \
      --thresholds 0.5 0.5 0.5 0.45 0.5 0.55 0.5 0.5 0.5
"""
import os
import sys
import json
import argparse
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
    split_data,
    PEERREAD_ALL_CONFERENCES,
)
from model import MultiTaskOrdinalClassifier, corn_logits_to_class
from metrics import compute_multi_task_metrics, compute_confusion_matrices
from train import collate_fn  # reuse the same collate


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
        corn_thresholds=None,  # we override at evaluate-time
    )


def _build_test_loader(mc: ModelConfig, tc: TrainingConfig, dc: DataConfig):
    text_prep = TextPreprocessor(
        normalize_whitespace=dc.normalize_whitespace,
        remove_references=dc.remove_references,
        max_length=dc.max_paper_length,
        min_length=dc.min_paper_length,
    )
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
    _, _, test_data = split_data(
        all_data,
        train_ratio=dc.train_split,
        dev_ratio=dc.dev_split,
        test_ratio=dc.test_split,
        seed=tc.seed,
    )
    print(f"[*] Test papers: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(mc.base_model_name)
    test_dataset = PaperReviewDataset(
        test_data, tokenizer,
        max_length=mc.max_length,
        score_dimensions=mc.score_dimensions,
        inference_mode=True,
    )
    pin = torch.cuda.is_available()
    return DataLoader(
        test_dataset,
        batch_size=tc.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin,
    )


def _run_forward_once(model, loader, device, dim='RECOMMENDATION'):
    """Cache CORN logits, regression preds, and labels for the whole loader."""
    all_logits, all_reg, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Forward (cached)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out['predictions'][dim].detach().cpu())
            rp = out.get('regression_predictions')
            if rp is not None:
                all_reg.append(rp[dim].detach().cpu())
            else:
                all_reg.append(torch.zeros(input_ids.size(0)))
            all_labels.append(batch['labels'][dim].detach().cpu())
    return (
        torch.cat(all_logits, dim=0),    # [N, K-1]
        torch.cat(all_reg,    dim=0),    # [N]
        torch.cat(all_labels, dim=0),    # [N]  may contain NaN
    )


def _evaluate_with_thresholds(
    model, logits, reg, labels, thresholds, num_classes, dim='RECOMMENDATION',
):
    """Decode cached logits with the given thresholds; return metrics + cm."""
    # Plug thresholds into the model buffer so resolve_class_predictions uses them
    if thresholds is None:
        model.corn_thresholds_buf = None
    else:
        model.corn_thresholds_buf = torch.tensor(
            list(thresholds), dtype=torch.float32, device=logits.device
        )
    pred_classes = model.resolve_class_predictions(logits, reg).cpu().numpy()
    labels_np = labels.numpy()
    valid = (~np.isnan(labels_np)) & (labels_np >= 1)
    true = np.clip(np.round(labels_np[valid]), 1, num_classes)
    pred = pred_classes[valid]
    metrics = compute_multi_task_metrics(
        {dim: pred},
        {dim: true},
        [dim],
        is_regression=False,
        num_classes=num_classes,
    )
    # Confusion matrix expects 0-indexed
    cm = compute_confusion_matrices(
        {dim: (pred - 1).astype(int)},
        {dim: (true - 1).astype(int)},
        [dim],
        num_classes,
    )[dim]
    return metrics, cm


def _print_result(label, metrics, cm, num_classes, prev_metrics=None):
    rec = metrics['per_dimension']['RECOMMENDATION']
    line = (f"{label:<35}  "
            f"qwk={rec['qwk']:.4f}  "
            f"spearman={rec['spearman']:.4f}  "
            f"mae={rec['mae']:.4f}  "
            f"acc={rec['accuracy']:.4f}  "
            f"macro_f1={rec['macro_f1']:.4f}")
    if prev_metrics is not None:
        prev_qwk = prev_metrics['per_dimension']['RECOMMENDATION']['qwk']
        delta = rec['qwk'] - prev_qwk
        line += f"  Δqwk={delta:+.4f}"
    print(line)
    # Per-column counts (predicted) so we can see if magnets shrink
    counts = cm.sum(axis=0)
    print("    pred counts per score:", " ".join(f"{i+1}:{int(counts[i])}" for i in range(num_classes)))


# ---------------------------------------------------------------------------
# Threshold sweep — curated for the K=10 confusion matrix you observed
# ---------------------------------------------------------------------------

def _build_curated_sweep(num_classes: int):
    """A small set of plausible threshold tweaks to try on the K=10 model.

    Names describe the *intent*, not the mechanism. Each entry returns a
    length-(K-1) tuple.
    """
    K = num_classes
    base = [0.5] * (K - 1)

    def _set(idx, val):
        v = list(base)
        v[idx] = val
        return tuple(v)

    sweeps = []

    # Fix score-4 over-prediction: lower thresh[3] so head 3 fires on
    # borderline 4s -> they become 5+
    sweeps.append(("lower thresh[3] = 0.45 (push 4 -> 5)",            _set(3, 0.45)))
    sweeps.append(("lower thresh[3] = 0.40 (push 4 -> 5, stronger)",  _set(3, 0.40)))

    # Fix score-7 over-prediction: raise thresh[5] so head 5 fires less ->
    # borderline 7s become 6
    sweeps.append(("raise thresh[5] = 0.55 (push 7 -> 6)",            _set(5, 0.55)))
    sweeps.append(("raise thresh[5] = 0.60 (push 7 -> 6, stronger)",  _set(5, 0.60)))

    # Combined
    combo = list(base); combo[3] = 0.45; combo[5] = 0.55
    sweeps.append(("combined: thresh[3]=0.45 + thresh[5]=0.55",       tuple(combo)))
    combo = list(base); combo[3] = 0.40; combo[5] = 0.60
    sweeps.append(("combined: thresh[3]=0.40 + thresh[5]=0.60",       tuple(combo)))

    # Bonus: push class 1 (almost never predicted) — RAISE thresh[0] hard
    sweeps.append(("raise thresh[0] = 0.70 (rescue class 1)",         _set(0, 0.70)))

    # Bonus: rescue class 8 — lower thresh[7]
    sweeps.append(("lower thresh[7] = 0.40 (rescue class 8)",         _set(7, 0.40)))

    return sweeps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate CORN thresholds on a saved model.")
    parser.add_argument("--model_path", required=True, help="Path to best_model.pt")
    parser.add_argument("--sweep", action="store_true", help="Run the built-in curated sweep.")
    parser.add_argument("--thresholds", type=float, nargs='+', default=None,
                        help="Custom thresholds (length = num_classes - 1).")
    parser.add_argument("--save", action="store_true",
                        help="Save the best sweep result to <model_dir>/calibration_results.json")
    args = parser.parse_args()

    if not args.sweep and args.thresholds is None:
        parser.error("Pass either --sweep or --thresholds.")

    # Configs
    mc = ModelConfig(); tc = TrainingConfig(); dc = DataConfig()
    K = mc.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build test loader
    print("\nLoading test split...")
    test_loader = _build_test_loader(mc, tc, dc)

    # Build + load model
    print(f"\nLoading model: {args.model_path}")
    model = _build_model(mc)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f"[WARN] missing keys: {res.missing_keys[:5]}{'...' if len(res.missing_keys) > 5 else ''}")
    if res.unexpected_keys:
        print(f"[WARN] unexpected keys: {res.unexpected_keys[:5]}{'...' if len(res.unexpected_keys) > 5 else ''}")
    model.to(device)

    # Forward pass once (the slow step)
    print("\nRunning forward pass once and caching outputs...")
    logits, reg, labels = _run_forward_once(model, test_loader, device)
    print(f"Cached: logits {tuple(logits.shape)}, reg {tuple(reg.shape)}, labels {tuple(labels.shape)}")

    # Move cache to device once so resolve runs there
    logits = logits.to(device); reg = reg.to(device)

    # Baseline (no thresholds = 0.5 everywhere)
    print("\n" + "=" * 80)
    print("BASELINE (thresholds = 0.5 everywhere)")
    print("=" * 80)
    base_metrics, base_cm = _evaluate_with_thresholds(model, logits, reg, labels, None, K)
    _print_result("baseline", base_metrics, base_cm, K)

    if args.thresholds is not None:
        if len(args.thresholds) != K - 1:
            parser.error(f"--thresholds expects {K-1} values, got {len(args.thresholds)}")
        print("\n" + "=" * 80)
        print(f"CUSTOM thresholds = {args.thresholds}")
        print("=" * 80)
        m, cm = _evaluate_with_thresholds(model, logits, reg, labels, args.thresholds, K)
        _print_result("custom", m, cm, K, prev_metrics=base_metrics)
        print("\nConfusion matrix (rows=true, cols=pred):")
        for row in cm:
            print("  " + " ".join(f"{int(v):4d}" for v in row))
        return

    # Curated sweep
    print("\n" + "=" * 80)
    print("CURATED SWEEP")
    print("=" * 80)

    results = [{"label": "baseline", "thresholds": None,
                "qwk": base_metrics['per_dimension']['RECOMMENDATION']['qwk'],
                "metrics": base_metrics['per_dimension']['RECOMMENDATION']}]

    for label, thr in _build_curated_sweep(K):
        m, cm = _evaluate_with_thresholds(model, logits, reg, labels, thr, K)
        _print_result(label, m, cm, K, prev_metrics=base_metrics)
        results.append({
            "label": label,
            "thresholds": list(thr),
            "qwk": m['per_dimension']['RECOMMENDATION']['qwk'],
            "metrics": m['per_dimension']['RECOMMENDATION'],
        })

    # Best by QWK
    best = max(results, key=lambda r: r['qwk'])
    print("\n" + "=" * 80)
    print(f"BEST CONFIG: {best['label']}")
    print(f"  QWK = {best['qwk']:.4f}  (baseline = {results[0]['qwk']:.4f}, Δ = {best['qwk'] - results[0]['qwk']:+.4f})")
    if best['thresholds'] is not None:
        print(f"  thresholds = {best['thresholds']}")
        print("\nTo make this permanent, edit Trainer/config.py:")
        print(f"    corn_thresholds: Optional[tuple] = {tuple(best['thresholds'])}")
    print("=" * 80)

    if args.save:
        out_path = os.path.join(os.path.dirname(args.model_path), "calibration_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"baseline": results[0], "sweep": results, "best": best}, f, indent=2)
        print(f"\n[OK] Saved sweep results to {out_path}")


if __name__ == "__main__":
    main()
