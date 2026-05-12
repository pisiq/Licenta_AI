"""
Fine-tune FLAN-T5 (seq2seq) to generate academic review text.

Architecture
------------
  Input  : TITLE + ABSTRACT + paper body excerpt + predicted scores
  Output : review text (the actual reviewer comments)

Training data comes from the entire `data/` folder (ACL 2017, CoNLL 2016,
ICLR 2017-2020, arXiv splits) via the existing load_peerread_data() pipeline.
Only samples that have non-empty review_comments are used.

Usage
-----
  # from the project root:
  python Trainer/review_generator_train.py

  # or with custom args:
  python Trainer/review_generator_train.py \
      --data_path data \
      --output_dir outputs/review_gen \
      --model_name google/flan-t5-base \
      --epochs 3 \
      --batch_size 4
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import math
from functools import partial
from typing import List, Dict, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure Trainer package is importable when run from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random as _random

from Trainer.config import ModelConfig as _ModelConfig, DataConfig as _DataConfig
from Trainer.data_preprocessing import (
    TextPreprocessor,
    load_peerread_data,
    split_data,
    PaperReview,
    SCORE_DIMENSIONS,
    PEERREAD_ALL_CONFERENCES,
    ICLR_CONFERENCES,
)
from Trainer.review_parser import parse_review, format_structured_target

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"   # scientific-domain T5
# Alternatives:
#   "google/flan-t5-base"             # instruction-tuned T5; non-scientific
#   "razent/SciFive-large-Pubmed_PMC" # larger SciFive; needs LoRA on 8GB
#   "google/flan-t5-small"            # ~80M; faster, lower quality

MAX_INPUT_TOKENS  = 1024   # truncate paper input
MAX_TARGET_TOKENS = 384    # max generated review length (structured output)
PAPER_BODY_CHARS  = 3_500  # chars of body text included in the prompt


# ===========================================================================
# Prompt builder
# ===========================================================================

def _scores_text(
    scores: Dict[str, Optional[float]],
    num_classes: int = 10,
    individual_scores: Optional[Dict[str, list]] = None,
) -> str:
    """Format the scores into a short readable string at the K=N scale.

    If `individual_scores` (per-reviewer) are present, append the spread
    (min/max across reviewers) to give the prompt more per-paper variability
    and signal reviewer disagreement.
    """
    parts = []
    for dim in SCORE_DIMENSIONS:
        val = scores.get(dim)
        if val is None:
            continue
        ind = (individual_scores or {}).get(dim) if individual_scores else None
        if ind:
            lo, hi = min(ind), max(ind)
            parts.append(f"{dim}: {val:.2f}/{num_classes} (reviewers ranged {lo:.0f}-{hi:.0f})")
        else:
            parts.append(f"{dim}: {val:.2f}/{num_classes}")
    return "  |  ".join(parts) if parts else "no scores"


def build_input_prompt(paper: PaperReview, num_classes: int = 10) -> str:
    """Build the seq2seq input prompt asking for a structured review.

    The prompt explicitly demands the output format (SUMMARY / STRENGTHS /
    WEAKNESSES / QUESTIONS), which combined with structured training
    targets prevents mode-collapse into freeform prose.
    """
    body_excerpt = paper.paper_text[:PAPER_BODY_CHARS] if paper.paper_text else ""
    scores_str   = _scores_text(
        paper.scores,
        num_classes=num_classes,
        individual_scores=paper.individual_scores,
    )
    venue = paper.conference if paper.conference else "unknown venue"
    return (
        f"Write a structured peer review for the following paper from {venue}. "
        f"Use this exact format (do not add other sections):\n"
        f"SUMMARY: <one paragraph summary>\n"
        f"STRENGTHS:\n- <strength bullet>\n- <strength bullet>\n"
        f"WEAKNESSES:\n- <weakness bullet>\n- <weakness bullet>\n"
        f"QUESTIONS:\n- <question for authors>\n\n"
        f"Title: {paper.title}\n\n"
        f"Abstract: {paper.abstract[:1500]}\n\n"
        f"Paper (excerpt): {body_excerpt}\n\n"
        f"Review scores: {scores_str}"
    )


# ===========================================================================
# Dataset
# ===========================================================================

class ReviewGenDataset(Dataset):
    """
    Each sample = (input_prompt, target_review_text).
    Only samples with non-empty review_comments are included.
    """

    def __init__(
        self,
        data: List[PaperReview],
        tokenizer,
        max_input_length:  int = MAX_INPUT_TOKENS,
        max_target_length: int = MAX_TARGET_TOKENS,
        min_review_chars:  int = 300,
        max_review_chars:  int = 30000,
        min_mean_confidence: float = 0.4,
        num_classes:       int = 10,
        keep_unparseable:  bool = True,
    ):
        self.tokenizer         = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length
        self.num_classes       = num_classes

        # (paper, pre-formatted structured target) tuples — we run the parser
        # ONCE at construction time so __getitem__ stays fast.
        self.samples: List = []
        n_short = n_long = n_lowconf = n_unparsed = n_freeform = 0
        for p in data:
            rc = (p.review_comments or "").strip()
            if not rc:
                continue
            if len(rc) < min_review_chars:
                n_short += 1
                continue
            if len(rc) > max_review_chars:
                n_long += 1
                continue
            if min_mean_confidence > 0 and p.individual_confidences:
                confs = [c for c in p.individual_confidences if c is not None]
                if confs and (sum(confs) / len(confs)) < min_mean_confidence:
                    n_lowconf += 1
                    continue
            parsed = parse_review(rc, allow_freeform=keep_unparseable)
            if parsed is None:
                n_unparsed += 1
                continue
            # Distinguish "had structure" from "freeform fallback" for stats
            if not parsed.get("strengths") and not parsed.get("weaknesses") and not parsed.get("questions"):
                n_freeform += 1
            target_text = format_structured_target(parsed)
            self.samples.append((p, target_text))

        n_in = len(data)
        n_out = len(self.samples)
        n_structured = n_out - n_freeform
        print(f"  ReviewGenDataset: {n_out}/{n_in} samples kept  "
              f"(structured={n_structured}, summary-only={n_freeform})")
        print(f"    dropped: <{min_review_chars}chars={n_short}, >{max_review_chars}chars={n_long}, "
              f"low_conf<{min_mean_confidence}={n_lowconf}, unparseable={n_unparsed}")
        if n_out == 0:
            print("  [WARN] zero samples remain. Try --no_keep_unparseable=False (it's True by default), "
                  "or lower --min_review_chars.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        paper, target = self.samples[idx]
        prompt = build_input_prompt(paper, num_classes=self.num_classes)

        # NO padding here — padding done dynamically in collate_fn per-batch
        enc = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        # Tokenise target without deprecated as_target_tokenizer()
        dec = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Dynamic padding — pad only to the longest sequence in the batch.
    This is the biggest speedup: avoids processing 1024 zeros for short samples."""
    input_ids      = pad_sequence([b["input_ids"]      for b in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels         = pad_sequence([b["labels"]         for b in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    model,
    train_loader:  DataLoader,
    dev_loader:    DataLoader,
    tokenizer,
    optimizer,
    scheduler,
    device:        torch.device,
    num_epochs:    int,
    output_dir:    str,
    precision:     str  = "bf16",     # "bf16" | "fp16" | "fp32"
    log_every:     int  = 50,
    grad_accum:    int  = 4,
):
    """
    T5 family models are unstable in fp16 (they were trained in bf16) — fp16
    autocast typically produces NaN losses partway through training. Default
    is bf16: same speed as fp16, no overflow, full T5 stability. Use fp32 if
    your GPU doesn't support bf16 (pre-Ampere).
    """
    # Resolve precision to autocast dtype + whether to use a GradScaler
    precision = precision.lower()
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"Unknown precision: {precision!r}")

    use_cuda = device.type == "cuda"
    use_autocast = use_cuda and precision != "fp32"
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    # GradScaler is only meaningful for fp16. bf16 has fp32's exponent range
    # so loss scaling is unnecessary (and gradient unscaling produces NaNs).
    needs_scaler = use_cuda and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    print(f"  Precision: {precision} (autocast={'on' if use_autocast else 'off'}, "
          f"grad_scaler={'on' if needs_scaler else 'off'})")

    def _optimizer_step_and_schedule() -> None:
        """Step optimizer, then scheduler, only if an optimizer step really happened."""
        if needs_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # On overflow, GradScaler skips optimizer.step(); don't advance LR.
            if scaler.get_scale() >= prev_scale:
                scheduler.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad()

    # torch.compile — requires Triton (Linux only). Skip on Windows.
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")
    else:
        print(f"  torch.compile: skipped (Windows — Triton not available)")

    best_dev_loss = float("inf")
    optimizer.zero_grad()

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")):
            # non_blocking=True overlaps CPU→GPU transfer with computation
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_autocast):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss / grad_accum   # normalize loss for accumulation

            if needs_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Only update every grad_accum steps
            if (step + 1) % grad_accum == 0:
                _optimizer_step_and_schedule()

            running_loss += loss.item() * grad_accum  # un-normalize for logging
            if (step + 1) % log_every == 0:
                avg = running_loss / (step + 1)
                print(f"  [Epoch {epoch}  step {step+1}]  train_loss={avg:.4f}")

        # Flush any remaining gradients at end of epoch
        if len(train_loader) % grad_accum != 0:
            _optimizer_step_and_schedule()

        train_loss = running_loss / len(train_loader)

        # ---- Eval ----
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{num_epochs} [dev]"):
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels         = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_autocast):
                    dev_loss += model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).loss.item()

        dev_loss /= len(dev_loader)
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}  |  train_loss={train_loss:.4f}  |  dev_loss={dev_loss:.4f}")
        print(f"{'='*60}\n")

        # ---- Save checkpoint ----
        ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved → {ckpt_dir}")

        # ---- Save best model ----
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dir = os.path.join(output_dir, "best_review_gen_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ★ New best model saved → {best_dir}  (dev_loss={dev_loss:.4f})")

    print(f"\nTraining complete. Best dev_loss={best_dev_loss:.4f}")
    print(f"Best model saved at: {os.path.join(output_dir, 'best_review_gen_model')}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune FLAN-T5 for review generation")
    p.add_argument("--data_path",   default="../data",
                   help="Path to the data folder (contains ICLR_*, acl_2017, etc.)")
    p.add_argument("--output_dir",  default="../outputs/review_gen",
                   help="Directory for checkpoints and best model")
    p.add_argument("--model_name",  default=DEFAULT_MODEL_NAME,
                   help="HuggingFace model name or local path for the base seq2seq model")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Per-device train batch size. Keep at 1 for 8GB VRAM.")
    p.add_argument("--grad_accum",  type=int,   default=16,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="bf16 (default, recommended for T5) | fp16 (UNSAFE — causes NaN on T5) | fp32")
    # Back-compat aliases:
    p.add_argument("--fp16",    dest="precision", action="store_const", const="fp16",
                   help="(deprecated) alias for --precision fp16. T5 is unstable in fp16.")
    p.add_argument("--no_fp16", dest="precision", action="store_const", const="fp32",
                   help="(deprecated) alias for --precision fp32.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max_input_tokens",  type=int, default=MAX_INPUT_TOKENS)
    p.add_argument("--max_target_tokens", type=int, default=MAX_TARGET_TOKENS)
    p.add_argument("--min_review_chars",  type=int, default=300,
                   help="Minimum length of review_comments to include a sample")
    p.add_argument("--max_review_chars",  type=int, default=30000,
                   help="Drop only obviously-spam reviews. Real ICLR reviews can be 8k-15k chars.")
    p.add_argument("--keep_unparseable",  action="store_true", default=True,
                   help="Keep reviews that don't parse into sections; format them as summary-only.")
    p.add_argument("--no_keep_unparseable", dest="keep_unparseable", action="store_false")
    p.add_argument("--min_mean_confidence", type=float, default=0.4,
                   help="Drop papers whose mean reviewer confidence is below this (0-1). 0 disables.")
    p.add_argument("--conferences", nargs="*", default=None,
                   help="Subset of conferences to load. Default = all.")
    p.add_argument("--iclr_only", action="store_true", default=True,
                   help="Restrict to ICLR conferences (the only ones with reviewer confidence).")
    p.add_argument("--no_iclr_only", dest="iclr_only", action="store_false")
    p.add_argument("--gen_dev_ratio", type=float, default=0.10,
                   help="Fraction of train_data to carve out as the generator's dev set.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Model base : {args.model_name}")
    print(f"Data path  : {args.data_path}")
    print(f"Output dir : {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading data...")
    preprocessor = TextPreprocessor(
        normalize_whitespace=True,
        remove_references=True,
        max_length=50_000,
        min_length=0,
    )
    # Pull data scale from the scoring model's config so the prompt format
    # matches what the API serves.
    _mc = _ModelConfig(); _dc = _DataConfig()
    num_classes  = _mc.num_classes
    target_scale = float(_dc.score_scale)

    all_data = load_peerread_data(
        base_data_path=args.data_path,
        text_preprocessor=preprocessor,
        conference_folders=args.conferences,   # None = all in PEERREAD_ALL_CONFERENCES
        require_pdf=True,
        verbose=True,
        seed=args.seed,
        target_scale=target_scale,
        iclr_only=args.iclr_only,
    )

    # split_data folds dev into test now → dev_data is always []. Carve a
    # generator-specific dev slice out of train so we still have a held-out
    # set for early-stop / best-checkpoint selection. The scoring model's
    # test set stays untouched — that's reserved for end-to-end evaluation.
    train_data, _, _test_data = split_data(all_data, seed=args.seed)
    rng = _random.Random(args.seed)
    rng.shuffle(train_data)
    n_dev = max(1, int(len(train_data) * float(args.gen_dev_ratio)))
    dev_data   = train_data[:n_dev]
    train_data = train_data[n_dev:]
    print(f"  train={len(train_data)}  dev(carved from train, ratio={args.gen_dev_ratio})={len(dev_data)}  "
          f"(scoring-model test set untouched: {len(_test_data)})")

    # -----------------------------------------------------------------------
    # 2. Load tokenizer & model
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # 3. Build datasets & loaders
    # -----------------------------------------------------------------------
    print("\n[3/4] Building datasets...")
    train_ds = ReviewGenDataset(
        train_data, tokenizer,
        max_input_length=args.max_input_tokens,
        max_target_length=args.max_target_tokens,
        min_review_chars=args.min_review_chars,
        max_review_chars=args.max_review_chars,
        min_mean_confidence=args.min_mean_confidence,
        num_classes=num_classes,
        keep_unparseable=args.keep_unparseable,
    )
    dev_ds = ReviewGenDataset(
        dev_data, tokenizer,
        max_input_length=args.max_input_tokens,
        max_target_length=args.max_target_tokens,
        min_review_chars=args.min_review_chars,
        max_review_chars=args.max_review_chars,
        min_mean_confidence=args.min_mean_confidence,
        num_classes=num_classes,
        keep_unparseable=args.keep_unparseable,
    )

    if len(train_ds) == 0:
        print("ERROR: No training samples found with non-empty review text!")
        print("       Check that your data/ folder is populated correctly.")
        sys.exit(1)

    # Dynamic collate_fn — pads only to longest in batch (huge speedup)
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    # num_workers>0 loads next batch in parallel while GPU trains current one
    _nw = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        prefetch_factor=2 if _nw > 0 else None,
        persistent_workers=_nw > 0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size * 4,   # eval can afford larger batch
        shuffle=False,
        num_workers=_nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        prefetch_factor=2 if _nw > 0 else None,
        persistent_workers=_nw > 0,
    )

    # -----------------------------------------------------------------------
    # 4. Optimizer + scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps   = updates_per_epoch * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"\n  Total optimizer steps : {total_steps}")
    print(f"  Warmup steps          : {warmup_steps}")

    # Save training config for reproducibility
    cfg = vars(args)
    cfg["train_samples"] = len(train_ds)
    cfg["dev_samples"]   = len(dev_ds)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # -----------------------------------------------------------------------
    # 5. Train
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Training for {args.epochs} epoch(s)...\n")
    # On CPU, force fp32 regardless of CLI flag — autocast bf16/fp16 is CUDA-only here.
    precision = args.precision if device.type == "cuda" else "fp32"

    train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        precision=precision,
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    main()

