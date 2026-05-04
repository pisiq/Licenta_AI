"""
Data preprocessing utilities for scientific paper review scoring.

Supported conferences
---------------------
ACL 2017 / CoNLL 2016
    • Full 8-score suite on a 1-5 scale.
    • Pre-defined train/dev/test sub-folders from PeerRead.
    • For training, dev is treated as test (no separate dev).

ICLR 2017-2020
    • Only RECOMMENDATION (rating string, numeric 1-10 → normalised to 1-5)
      and REVIEWER_CONFIDENCE (confidence string, numeric 1-5).
    • Flat layout: <conf>/reviews/ and <conf>/parsed_pdfs/ with NO split folders.
    • Automatic 80/20 random split applied per ICLR year (dev == test).

Task design
-----------
Primary target  : RECOMMENDATION  (the score we care about most)
Auxiliary targets: the other 7 dimensions — used as weighted multi-task
                  signals for ACL/CoNLL.  ICLR samples have these masked out
                  (score_mask=False) so the loss ignores them.

Loss weights (used in model.py / trainer.py)
    RECOMMENDATION        : 3.0   (primary)
    REVIEWER_CONFIDENCE   : 0.3
    all others            : 0.5
"""
import os
import re
import glob
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Canonical score dimensions (PRIMARY first, then auxiliaries)
# ---------------------------------------------------------------------------
SCORE_DIMENSIONS: List[str] = [
    "RECOMMENDATION",           # PRIMARY — all conferences
    "IMPACT",                   # auxiliary — ACL / CoNLL only
    "SUBSTANCE",                # auxiliary — ACL / CoNLL only
    "APPROPRIATENESS",          # auxiliary — ACL / CoNLL only
    "MEANINGFUL_COMPARISON",    # auxiliary — ACL / CoNLL only
    "SOUNDNESS_CORRECTNESS",    # auxiliary — ACL / CoNLL only
    "ORIGINALITY",              # auxiliary — ACL / CoNLL only
    "CLARITY",                  # auxiliary — ACL / CoNLL only
    # NOTE: REVIEWER_CONFIDENCE lives inside the review but is NOT a
    # paper-quality score so we intentionally exclude it from
    # SCORE_DIMENSIONS.  This keeps the output head at 7+1 = 8 and avoids
    # conflating reviewer confidence with paper quality.
]

# Per-dimension loss weights (used by model.py)
SCORE_WEIGHTS: Dict[str, float] = {
    "RECOMMENDATION":        3.0,
    "IMPACT":                0.5,
    "SUBSTANCE":             0.5,
    "APPROPRIATENESS":       0.5,
    "MEANINGFUL_COMPARISON": 0.5,
    "SOUNDNESS_CORRECTNESS": 0.5,
    "ORIGINALITY":           0.5,
    "CLARITY":               0.5,
}

# ---------------------------------------------------------------------------
# Conference definitions
# ---------------------------------------------------------------------------

# Conferences with pre-defined PeerRead train/dev/test splits
PEERREAD_SPLIT_CONFERENCES: List[str] = ["acl_2017", "conll_2016"]

# ICLR conferences — flat layout, we auto-split
ICLR_CONFERENCES: List[str] = ["ICLR_2017", "ICLR_2018", "ICLR_2019", "ICLR_2020"]

# Default: use ACL + CoNLL + all ICLR years
PEERREAD_ALL_CONFERENCES: List[str] = (
    PEERREAD_SPLIT_CONFERENCES + ICLR_CONFERENCES
)

# Kept for backward compat
PEERREAD_SCORED_CONFERENCES: List[str] = PEERREAD_ALL_CONFERENCES

# Max raw score per conference (used for normalisation)
_CONFERENCE_SCORE_MAX: Dict[str, float] = {
    "acl_2017":   5.0,
    "conll_2016": 5.0,
    "ICLR_2017":  10.0,
    "ICLR_2018":  10.0,
    "ICLR_2019":  10.0,
    "ICLR_2020":  10.0,
}


# ===========================================================================
# Core data class
# ===========================================================================

@dataclass
class PaperReview:
    """Data structure for a paper paired with its review scores."""
    paper_id:        str
    conference:      str
    split:           str          # "train" / "dev" / "test"
    title:           str
    abstract:        str
    paper_text:      str          # body sections from parsed PDF
    review_comments: str          # concatenated reviewer comments
    combined_text:   str          # PAPER [SEP] REVIEW (used during training)
    scores:          Dict[str, Optional[float]]   # dim -> mean score or None
    score_mask:      Dict[str, bool]              # dim -> True if valid

    @property
    def full_text(self) -> str:
        """Legacy alias."""
        return self.paper_text


# ===========================================================================
# Text preprocessing
# ===========================================================================

class TextPreprocessor:
    """Cleans and normalises scientific paper text."""

    def __init__(self,
                 normalize_whitespace: bool = True,
                 remove_references:    bool = True,
                 max_length:           int  = 10_000,
                 min_length:           int  = 100):
        self.normalize_whitespace = normalize_whitespace
        self.remove_references    = remove_references
        self.max_length           = max_length
        self.min_length           = min_length

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\x00', '', text)
        text = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'- ', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def remove_references_section(self, text: str) -> str:
        if not self.remove_references:
            return text
        for pat in [r'\n\s*REFERENCES\s*\n', r'\n\s*References\s*\n',
                    r'\n\s*BIBLIOGRAPHY\s*\n', r'\n\s*Bibliography\s*\n']:
            m = re.search(pat, text)
            if m:
                text = text[:m.start()]
                break
        return text

    def truncate_text(self, text: str) -> str:
        return text[:self.max_length] if len(text) > self.max_length else text

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        text = self.remove_references_section(text)
        text = self.truncate_text(text)
        return text


# ===========================================================================
# Review aggregator (legacy helper kept for API compat)
# ===========================================================================

class ReviewAggregator:
    def __init__(self, method: str = "mean_round", min_val: int = 1, max_val: int = 5):
        self.method  = method
        self.min_val = min_val
        self.max_val = max_val

    def aggregate_scores(self, reviews: List[Dict[str, Any]]) -> Dict[str, int]:
        if len(reviews) == 1:
            return reviews[0]
        dim_scores: Dict[str, List] = {}
        for rv in reviews:
            for dim, score in rv.items():
                dim_scores.setdefault(dim, []).append(score)
        aggregated = {}
        for dim, scores in dim_scores.items():
            val = (int(np.round(np.mean(scores))) if self.method == "mean_round"
                   else int(np.median(scores)))
            aggregated[dim] = int(np.clip(val, self.min_val, self.max_val))
        return aggregated


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class PaperReviewDataset(Dataset):
    """
    Training mode   (inference_mode=False):
        Input = TITLE + ABSTRACT + PAPER body + [SEP] + REVIEW comments
    Inference mode  (inference_mode=True):
        Input = TITLE + ABSTRACT + PAPER body   (no review leakage)
    """

    def __init__(self,
                 data:             List[PaperReview],
                 tokenizer,
                 max_length:       int  = 4096,
                 score_dimensions: List[str] = None,
                 print_summary:    bool = True,
                 inference_mode:   bool = False):
        self.data             = data
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.score_dimensions = score_dimensions or SCORE_DIMENSIONS
        self.inference_mode   = inference_mode

        if print_summary and data:
            n       = len(data)
            tag     = "[INFERENCE]" if inference_mode else "[TRAINING]"
            print(f"\n[OK] {n} samples loaded. {tag}")
            for dim in self.score_dimensions:
                valid = sum(1 for p in data if p.score_mask.get(dim, False))
                print(f"   {dim:<30}: {valid} valid")

    def _encode(self, text: str) -> List[int]:
        """Tokenise + encode a single text string."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _pad(self, encodings: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
        """Pad sequences to max_length, return array + mask."""
        # NOTE: we could move this to a collate_fn but then we'd need to pad
        #       each time in the DataLoader, which is slower and messes up
        #       batch statistics (e.g. for gradient accumulation).
        #       So we pre-pad here to the fixed max_length.
        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.mask_token_id if pad_id != 0 else 0
        # Pad sequences to max_length
        padded = [e + [pad_id] * (self.max_length - len(e)) for e in encodings]
        # Compute attention masks (1 for real tokens, 0 for padding)
        masks = [[1 if t != pad_id else 0 for t in seq] for seq in padded]
        return np.array(padded, dtype=np.int64), np.array(masks, dtype=np.int64)

    def _truncate(self, encodings: List[List[int]]) -> List[List[int]]:
        """Truncate sequences to max_length."""
        return [e[:self.max_length] for e in encodings]

    def _prepare_sample(self, sample: PaperReview) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert a single sample to model inputs."""
        # Encode paper text (title + abstract + body)
        enc_paper = self._encode(sample.title + " " + sample.abstract + " " + sample.paper_text)
        # Encode review comments (target)
        enc_review = self._encode(sample.review_comments)

        # Truncate if too long
        if len(enc_paper) + len(enc_review) > self.max_length:
            enc_paper = self._truncate(enc_paper)
            enc_review = self._truncate(enc_review)

        # Pad to max_length
        enc_paper, mask_paper = self._pad([enc_paper])
        enc_review, mask_review = self._pad([enc_review])

        # Concatenate paper and review encodings
        input_ids = np.concatenate([enc_paper, enc_review], axis=1)
        attention_mask = np.concatenate([mask_paper, mask_review], axis=1)

        # Build score tensor (multi-dimensional, one hot per dimension)
        score_tensor = np.zeros((len(SCORE_DIMENSIONS), self.max_length), dtype=np.float32)
        for i, dim in enumerate(SCORE_DIMENSIONS):
            if sample.score_mask.get(dim, False):
                score_tensor[i, :len(enc_review[0])] = sample.scores[dim]

        return input_ids.flatten(), attention_mask.flatten(), enc_review.flatten(), score_tensor.flatten()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get item for PyTorch."""
        sample = self.data[idx]
        return self._prepare_sample(sample)

    def __len__(self) -> int:
        """Dataset size."""
        return len(self.data)


# ===========================================================================
# Data loading
# ===========================================================================

def split_data(
    data:        List[PaperReview],
    train_ratio: float = 0.8,
    dev_ratio:   float = 0.1,
    test_ratio:  float = 0.1,
    seed:        int   = 42,
) -> Tuple[List[PaperReview], List[PaperReview], List[PaperReview]]:
    """
    Smart splitter:
      • If the data already has valid split labels (from pre-defined PeerRead
        folders or from the ICLR auto-split), use them directly.
      • Otherwise fall back to a random shuffle.

    If dev_ratio == 0, dev is treated as test (no separate dev set).
    """
    has_predef = any(s.split in ("train", "dev", "test") for s in data)
    train_has  = any(s.split == "train" for s in data)
    dev_has    = any(s.split == "dev"   for s in data)
    test_has   = any(s.split == "test"  for s in data)

    if has_predef and train_has and dev_has:
        # Use pre-defined splits; treat dev as test (no separate dev)
        train = [s for s in data if s.split == "train" or s.split == "test"]
        test  = [s for s in data if s.split == "dev"]
        dev   = test
        return train, dev, test

    # Random fallback
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    n_train = int(len(data) * train_ratio)
    n_dev   = int(len(data) * dev_ratio)

    if n_dev == 0:
        train = [data[i] for i in indices[:n_train]]
        test  = [data[i] for i in indices[n_train:]]
        dev   = test
    else:
        train = [data[i] for i in indices[:n_train]]
        dev   = [data[i] for i in indices[n_train:n_train + n_dev]]
        test  = [data[i] for i in indices[n_train + n_dev:]]

    return train, dev, test

