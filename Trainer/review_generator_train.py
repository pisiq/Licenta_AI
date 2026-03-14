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

from Trainer.data_preprocessing import (
    TextPreprocessor,
    load_peerread_data,
    split_data,
    PaperReview,
    SCORE_DIMENSIONS,
)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "google/flan-t5-base"   # ~250M params, fits 8 GB VRAM
# Alternatives:
#   "google/flan-t5-small"   ~80M  – faster, less quality
#   "google/flan-t5-large"  ~780M  – better, needs ~14 GB VRAM

MAX_INPUT_TOKENS  = 1024   # truncate paper input
MAX_TARGET_TOKENS = 512    # max generated review length
PAPER_BODY_CHARS  = 3_500  # chars of body text included in the prompt


# ===========================================================================
# Prompt builder
# ===========================================================================

def _scores_text(scores: Dict[str, Optional[float]]) -> str:
    """Format the predicted/actual scores into a short readable string."""
    lines = []
    for dim in SCORE_DIMENSIONS:
        val = scores.get(dim)
        if val is not None:
            lines.append(f"{dim}: {val:.2f}/5.0")
    return "  |  ".join(lines) if lines else "no scores"


def build_input_prompt(paper: PaperReview) -> str:
    """
    Build the seq2seq input prompt from a PaperReview object.
    """
    body_excerpt = paper.paper_text[:PAPER_BODY_CHARS] if paper.paper_text else ""
    scores_str   = _scores_text(paper.scores)
    return (
        f"Write an academic peer review for the following machine learning paper.\n\n"
        f"Title: {paper.title}\n\n"
        f"Abstract: {paper.abstract[:1500]}\n\n"
        f"Paper (excerpt): {body_excerpt}\n\n"
        f"Review scores: {scores_str}\n\n"
        f"Write a detailed review covering: summary, strengths, weaknesses, "
        f"questions for authors, and final recommendation."
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
        min_review_chars:  int = 100,
    ):
        self.samples = [
            p for p in data
            if p.review_comments and len(p.review_comments.strip()) >= min_review_chars
        ]
        self.tokenizer         = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length

        print(f"  ReviewGenDataset: {len(self.samples)} samples "
              f"(out of {len(data)} total, filtered by review length >= {min_review_chars} chars)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        paper  = self.samples[idx]
        prompt = build_input_prompt(paper)
        target = paper.review_comments.strip()

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
    fp16:          bool = True,
    log_every:     int  = 50,
    grad_accum:    int  = 4,
):
    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _optimizer_step_and_schedule() -> None:
        """Step optimizer, then scheduler, only if optimizer step really happened."""
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if use_amp:
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # On overflow, GradScaler skips optimizer.step(); do not advance LR then.
            if scaler.get_scale() >= prev_scale:
                scheduler.step()
        else:
            scaler.step(optimizer)
            scaler.update()
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

            with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss / grad_accum   # normalize loss for accumulation

            scaler.scale(loss).backward()

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
                with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
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
    p.add_argument("--batch_size",  type=int,   default=4,
                   help="Per-device train batch size")
    p.add_argument("--grad_accum",  type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--no_fp16",     dest="fp16", action="store_false")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max_input_tokens",  type=int, default=MAX_INPUT_TOKENS)
    p.add_argument("--max_target_tokens", type=int, default=MAX_TARGET_TOKENS)
    p.add_argument("--min_review_chars",  type=int, default=100,
                   help="Minimum length of review_comments to include a sample")
    p.add_argument("--conferences", nargs="*", default=None,
                   help="Subset of conferences to load. Default = all.")
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
    all_data = load_peerread_data(
        base_data_path=args.data_path,
        text_preprocessor=preprocessor,
        conference_folders=args.conferences,   # None = all conferences
        require_pdf=True,
        min_body_length=100,
        verbose=True,
    )

    # Use pre-defined splits
    train_data, dev_data, _ = split_data(all_data, seed=args.seed)
    print(f"  train={len(train_data)}  dev={len(dev_data)}")

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
    )
    dev_ds = ReviewGenDataset(
        dev_data, tokenizer,
        max_input_length=args.max_input_tokens,
        max_target_length=args.max_target_tokens,
        min_review_chars=args.min_review_chars,
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
        fp16=args.fp16 and device.type == "cuda",
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    main()

