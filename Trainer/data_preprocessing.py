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
Primary target  : RECOMMENDATION (the only score used for training)
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
# Canonical score dimensions (PRIMARY only)
# ---------------------------------------------------------------------------
SCORE_DIMENSIONS: List[str] = [
    "RECOMMENDATION",
]

# Per-dimension loss weights (used by model.py)
SCORE_WEIGHTS: Dict[str, float] = {
    "RECOMMENDATION": 1.0,
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
    paper_id:          str
    conference:        str
    split:             str          # "train" / "dev" / "test"
    title:             str
    abstract:          str
    paper_text:        str          # body sections from parsed PDF
    review_comments:   str          # concatenated reviewer comments
    combined_text:     str          # PAPER [SEP] REVIEW (used during training)
    scores:            Dict[str, Optional[float]]   # dim -> mean score or None
    score_mask:        Dict[str, bool]              # dim -> True if valid
    # Per-reviewer scores (normalized to the active target scale). Populated by
    # the loaders. None for legacy callers; safe to leave unset.
    individual_scores: Optional[Dict[str, List[float]]] = None
    # Per-reviewer confidences in [0.2, 1.0] = confidence/5.
    # Aligned positionally with `individual_scores[primary_dim]`.
    individual_confidences: Optional[List[Optional[float]]] = None

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

    def __init__(
        self,
        data: List[PaperReview],
        tokenizer,
        max_length: int = 4096,
        score_dimensions: List[str] = None,
        print_summary: bool = True,
        inference_mode: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_dimensions = score_dimensions or SCORE_DIMENSIONS
        self.inference_mode = inference_mode

        if print_summary and data:
            n = len(data)
            tag = "[INFERENCE]" if inference_mode else "[TRAINING]"
            print(f"\n[OK] {n} samples loaded. {tag}")
            for dim in self.score_dimensions:
                valid = sum(1 for p in data if p.score_mask.get(dim, False))
                print(f"   {dim:<30}: {valid:>4}/{n}  ({100*valid//n}%)")
            print()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch
        paper = self.data[idx]

        if self.inference_mode:
            input_text = _build_paper_only_text(
                paper.title, paper.abstract, paper.paper_text
            )
        else:
            input_text = paper.combined_text

        if not isinstance(input_text, str):
            if isinstance(input_text, (list, tuple)):
                input_text = " ".join(str(x) for x in input_text)
            else:
                input_text = str(input_text) if input_text is not None else ""

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels: Dict[str, Any] = {}
        label_mask: Dict[str, Any] = {}
        for dim in self.score_dimensions:
            score = paper.scores.get(dim)
            valid = paper.score_mask.get(dim, False) and score is not None
            labels[dim] = torch.tensor(float(score) if valid else float("nan"), dtype=torch.float32)
            label_mask[dim] = torch.tensor(1.0 if valid else 0.0, dtype=torch.float32)

        # Per-sample confidence weight in (0, 1]. Defaults to 1.0 when no
        # per-reviewer confidence is available. Post-expansion samples carry
        # exactly one entry in `individual_confidences`.
        conf_w = 1.0
        ind_c = paper.individual_confidences
        if ind_c is not None and len(ind_c) >= 1 and ind_c[0] is not None:
            conf_w = float(ind_c[0])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "label_mask": label_mask,
            "confidence_weight": torch.tensor(conf_w, dtype=torch.float32),
        }


def _build_paper_only_text(title: str, abstract: str, body_text: str) -> str:
    title = title or ""
    abstract = abstract or ""
    body_text = body_text or ""
    return f"TITLE: {title}\n\nABSTRACT: {abstract}\n\n{body_text}".strip()


def _build_combined_text(title: str, abstract: str, body_text: str, review_comments: str) -> str:
    paper_text = _build_paper_only_text(title, abstract, body_text)
    review_comments = review_comments or ""
    return f"{paper_text}\n\n[REVIEW]\n{review_comments}".strip()


def _parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        return float(m.group(0)) if m else None
    return None


def _normalize_score(value: Optional[float], conf: str, target_scale: float = 5.0) -> Optional[float]:
    """Normalize a raw rating to the target scale (default 5; pass 10 for native ICLR)."""
    if value is None:
        return None
    max_val = _CONFERENCE_SCORE_MAX.get(conf, 5.0)
    if max_val != target_scale:
        value = value / max_val * target_scale
    return float(np.clip(value, 1.0, target_scale))


# Reviewer-confidence parsing. ICLR stores integers 1-5 (sometimes as
# strings like "4: The reviewer is confident..."). Returns the normalized
# weight in (0, 1] — confidence/5 — or None if missing.
_CONFIDENCE_MAX = 5.0

def _parse_confidence(rv: Dict[str, Any]) -> Optional[float]:
    raw = rv.get("confidence")
    val = _parse_numeric(raw)
    if val is None:
        return None
    val = float(np.clip(val, 1.0, _CONFIDENCE_MAX))
    return val / _CONFIDENCE_MAX  # -> [0.2, 1.0]


def _load_parsed_pdf_text(parsed_path: str) -> Tuple[str, str, str]:
    try:
        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "", "", ""

    metadata = data.get("metadata", {})
    title = metadata.get("title") or data.get("title") or ""
    abstract = metadata.get("abstract") or data.get("abstractText") or ""
    sections = metadata.get("sections", []) or data.get("sections", [])
    section_texts = [s.get("text", "") for s in sections if isinstance(s, dict)]
    body_text = "\n".join([t for t in section_texts if t])
    return title, abstract, body_text


def _aggregate_scores(
    reviews: List[Dict[str, Any]],
    conf: str,
    target_scale: float = 5.0,
) -> Tuple[Dict[str, Optional[float]], Dict[str, bool], Dict[str, List[float]]]:
    """Returns (mean_scores, score_mask, individual_scores) per dimension.

    Scores are normalized to `target_scale`. The third element preserves each
    reviewer's normalized score positionally so callers can replicate samples.
    """
    dim_values: Dict[str, List[float]] = {dim: [] for dim in SCORE_DIMENSIONS}
    for rv in reviews:
        for dim in SCORE_DIMENSIONS:
            if dim in rv:
                val = _parse_numeric(rv.get(dim))
                val = _normalize_score(val, conf, target_scale=target_scale)
                if val is not None:
                    dim_values[dim].append(val)

    scores: Dict[str, Optional[float]] = {}
    score_mask: Dict[str, bool] = {}
    for dim, vals in dim_values.items():
        if vals:
            scores[dim] = float(np.mean(vals))
            score_mask[dim] = True
        else:
            scores[dim] = None
            score_mask[dim] = False
    return scores, score_mask, dim_values


def load_peerread_data(
    base_data_path: str,
    text_preprocessor: TextPreprocessor,
    conference_folders: Optional[List[str]] = None,
    require_pdf: bool = True,
    verbose: bool = False,
    seed: int = 42,
    target_scale: float = 5.0,
    iclr_only: bool = False,
) -> List[PaperReview]:
    """Load PeerRead-style data from ACL/CoNLL (split) and ICLR (flat) folders.

    target_scale : 5.0 (PeerRead-normalized) or 10.0 (native ICLR) ratings.
    iclr_only    : if True, restrict to ICLR conferences only.
    """
    conference_folders = conference_folders or PEERREAD_ALL_CONFERENCES
    if iclr_only:
        conference_folders = [c for c in conference_folders if c in ICLR_CONFERENCES]
        if verbose:
            print(f"  [iclr_only=True] Loading only: {conference_folders}")
    rng = random.Random(seed)
    processed: List[PaperReview] = []

    for conf in conference_folders:
        conf_path = os.path.join(base_data_path, conf)
        if not os.path.isdir(conf_path):
            if verbose:
                print(f"  [SKIP] {conf} — folder not found at {conf_path}")
            continue

        if conf in PEERREAD_SPLIT_CONFERENCES:
            for split in ["train", "dev", "test"]:
                split_reviews = os.path.join(conf_path, split, "reviews")
                split_parsed = os.path.join(conf_path, split, "parsed_pdfs")
                if not os.path.isdir(split_reviews):
                    if verbose:
                        print(f"  [SKIP] {conf}/{split} — reviews folder not found")
                    continue

                for review_file in glob.glob(os.path.join(split_reviews, "*.json")):
                    paper_id = os.path.splitext(os.path.basename(review_file))[0]
                    parsed_path = os.path.join(split_parsed, f"{paper_id}.pdf.json")
                    if require_pdf and not os.path.isfile(parsed_path):
                        continue

                    try:
                        with open(review_file, "r", encoding="utf-8") as f:
                            review_data = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        continue

                    title = review_data.get("title") or ""
                    abstract = review_data.get("abstract") or ""
                    if os.path.isfile(parsed_path):
                        pdf_title, pdf_abs, body_text = _load_parsed_pdf_text(parsed_path)
                        title = title or pdf_title
                        abstract = abstract or pdf_abs
                    else:
                        body_text = ""

                    reviews = review_data.get("reviews", [])
                    review_comments = "\n\n".join([
                        r.get("comments", "") for r in reviews if isinstance(r, dict) and r.get("comments")
                    ])

                    scores, score_mask, individual = _aggregate_scores(reviews, conf, target_scale=target_scale)
                    if not any(score_mask.values()):
                        continue
                    # ACL/CoNLL reviews don't carry a confidence field — leave None.
                    individual_conf = [None] * len(individual.get("RECOMMENDATION", []))

                    title = text_preprocessor.preprocess(title)
                    abstract = text_preprocessor.preprocess(abstract)
                    paper_text = text_preprocessor.preprocess(body_text)
                    review_comments = text_preprocessor.preprocess(review_comments)

                    processed.append(PaperReview(
                        paper_id=paper_id,
                        conference=conf,
                        split=split,
                        title=title,
                        abstract=abstract,
                        paper_text=paper_text,
                        review_comments=review_comments,
                        combined_text=_build_combined_text(title, abstract, paper_text, review_comments),
                        scores=scores,
                        score_mask=score_mask,
                        individual_scores=individual,
                        individual_confidences=individual_conf,
                    ))

        elif conf in ICLR_CONFERENCES:
            review_dir = os.path.join(conf_path, "reviews")
            parsed_dir = os.path.join(conf_path, "parsed_pdfs")
            if not os.path.isdir(review_dir):
                if verbose:
                    print(f"  [SKIP] {conf} — reviews folder not found")
                continue

            review_files = sorted(glob.glob(os.path.join(review_dir, "*.json")))
            rng.shuffle(review_files)
            split_point = int(len(review_files) * 0.8)
            train_set = set(review_files[:split_point])

            for review_file in review_files:
                try:
                    with open(review_file, "r", encoding="utf-8") as f:
                        review_data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue

                paper_id = review_data.get("id") or os.path.splitext(os.path.basename(review_file))[0].replace("_review", "")
                parsed_path = os.path.join(parsed_dir, f"{paper_id}_content.json")
                if require_pdf and not os.path.isfile(parsed_path):
                    continue

                title, abstract, body_text = _load_parsed_pdf_text(parsed_path)
                reviews = review_data.get("reviews", [])

                review_comments = "\n\n".join([
                    r.get("review", "") for r in reviews if isinstance(r, dict) and r.get("review")
                ])

                rec_scores: List[float] = []
                rec_confs: List[Optional[float]] = []
                for rv in reviews:
                    rating = _parse_numeric(rv.get("rating"))
                    rating = _normalize_score(rating, conf, target_scale=target_scale)
                    if rating is not None:
                        rec_scores.append(rating)
                        rec_confs.append(_parse_confidence(rv))   # may be None

                scores = {dim: None for dim in SCORE_DIMENSIONS}
                score_mask = {dim: False for dim in SCORE_DIMENSIONS}
                individual: Dict[str, List[float]] = {dim: [] for dim in SCORE_DIMENSIONS}
                if rec_scores:
                    scores["RECOMMENDATION"] = float(np.mean(rec_scores))
                    score_mask["RECOMMENDATION"] = True
                    individual["RECOMMENDATION"] = list(rec_scores)

                if not any(score_mask.values()):
                    continue

                split = "train" if review_file in train_set else "dev"

                processed.append(PaperReview(
                    paper_id=paper_id,
                    conference=conf,
                    split=split,
                    title=title,
                    abstract=abstract,
                    paper_text=paper_text,
                    review_comments=review_comments,
                    combined_text=_build_combined_text(title, abstract, paper_text, review_comments),
                    scores=scores,
                    score_mask=score_mask,
                    individual_scores=individual,
                    individual_confidences=rec_confs,
                ))
        else:
            if verbose:
                print(f"  [SKIP] {conf} — unknown conference")

    if verbose:
        print(f"[OK] Loaded {len(processed)} papers total")
    return processed


def load_and_preprocess_data(
    data_path: str,
    text_preprocessor: TextPreprocessor,
    review_aggregator: ReviewAggregator,
) -> List[PaperReview]:
    """Load a single JSON file into PaperReview objects."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        papers = raw.get("papers") or raw.get("data") or [raw]
    else:
        papers = raw

    processed: List[PaperReview] = []
    for idx, item in enumerate(papers):
        if not isinstance(item, dict):
            continue

        title = item.get("title") or ""
        abstract = item.get("abstract") or ""
        paper_text = item.get("paper_text") or item.get("text") or item.get("paper") or ""
        reviews = item.get("reviews") or []

        review_comments = "\n\n".join([
            r.get("comments", "") for r in reviews if isinstance(r, dict) and r.get("comments")
        ])

        scores_raw: Dict[str, List[float]] = {dim: [] for dim in SCORE_DIMENSIONS}
        for rv in reviews:
            for dim in SCORE_DIMENSIONS:
                if dim in rv:
                    val = _parse_numeric(rv.get(dim))
                    if val is not None:
                        scores_raw[dim].append(val)

        scores: Dict[str, Optional[float]] = {}
        score_mask: Dict[str, bool] = {}
        for dim, vals in scores_raw.items():
            if vals:
                scores[dim] = float(np.mean(vals))
                score_mask[dim] = True
            else:
                scores[dim] = None
                score_mask[dim] = False

        if not any(score_mask.values()):
            continue

        title = text_preprocessor.preprocess(title)
        abstract = text_preprocessor.preprocess(abstract)
        paper_text = text_preprocessor.preprocess(paper_text)
        review_comments = text_preprocessor.preprocess(review_comments)
        combined_text = _build_combined_text(title, abstract, paper_text, review_comments)

        processed.append(PaperReview(
            paper_id=str(item.get("id") or idx),
            conference="legacy",
            split="train",
            title=title,
            abstract=abstract,
            paper_text=paper_text,
            review_comments=review_comments,
            combined_text=combined_text,
            scores=scores,
            score_mask=score_mask,
            individual_scores=scores_raw,
        ))

    return processed


def expand_per_reviewer(
    papers: List[PaperReview],
    primary_dim: str = "RECOMMENDATION",
) -> List[PaperReview]:
    """Replace each paper with N copies — one per individual reviewer score on
    `primary_dim`. Other dimensions stay at their mean. Papers without
    individual scores fall back to a single (mean-labeled) sample.

    Use this on the TRAIN split only. The test split should keep one
    sample per paper using the mean label.
    """
    expanded: List[PaperReview] = []
    skipped = 0
    for p in papers:
        ind = (p.individual_scores or {}).get(primary_dim) if p.individual_scores else None
        if not ind:
            # No per-reviewer info — keep the paper as-is.
            expanded.append(p)
            skipped += 1
            continue

        confs = p.individual_confidences or [None] * len(ind)
        # Defensive: pad/truncate if lengths got out of sync
        if len(confs) < len(ind):
            confs = list(confs) + [None] * (len(ind) - len(confs))
        elif len(confs) > len(ind):
            confs = list(confs)[:len(ind)]

        for s, c in zip(ind, confs):
            new_scores = dict(p.scores)
            new_mask = dict(p.score_mask)
            new_scores[primary_dim] = float(s)
            new_mask[primary_dim] = True
            expanded.append(PaperReview(
                paper_id=f"{p.paper_id}#r{len(expanded)}",
                conference=p.conference,
                split=p.split,
                title=p.title,
                abstract=p.abstract,
                paper_text=p.paper_text,
                review_comments=p.review_comments,
                combined_text=p.combined_text,
                scores=new_scores,
                score_mask=new_mask,
                individual_scores=p.individual_scores,
                # Single confidence value for THIS replicated sample.
                individual_confidences=[c],
            ))
    print(f"[expand_per_reviewer] {len(papers)} papers -> {len(expanded)} samples "
          f"(skipped {skipped} without per-reviewer info)")
    return expanded


def subsample_by_class(
    papers: List[PaperReview],
    cap: int,
    primary_dim: str = "RECOMMENDATION",
    num_classes: int = 5,
    seed: int = 42,
) -> List[PaperReview]:
    """Randomly subsample classes that exceed `cap` down to exactly `cap`
    samples. Classes at or below `cap` are kept untouched. Samples missing
    a label on `primary_dim` are kept (they may carry signal for aux dims).

    Use on the TRAIN split only.
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    no_label_idx: List[int] = []

    for i, p in enumerate(papers):
        score = p.scores.get(primary_dim) if p.scores else None
        valid = (
            p.score_mask.get(primary_dim, False)
            and score is not None
            and not (isinstance(score, float) and np.isnan(score))
        )
        if not valid:
            no_label_idx.append(i)
            continue
        c = int(np.clip(np.round(float(score)), 1, num_classes)) - 1
        by_class[c].append(i)

    keep_idx: List[int] = list(no_label_idx)
    summary_before = []
    summary_after = []
    for c in range(num_classes):
        bucket = by_class[c]
        before = len(bucket)
        if before > cap:
            rng.shuffle(bucket)
            kept = bucket[:cap]
        else:
            kept = bucket
        keep_idx.extend(kept)
        summary_before.append(before)
        summary_after.append(len(kept))

    keep_idx.sort()  # preserve original order
    out = [papers[i] for i in keep_idx]
    print(f"[subsample_by_class] cap={cap} | "
          f"per-class before {summary_before} -> after {summary_after} "
          f"(no-label kept: {len(no_label_idx)}) | total {len(papers)} -> {len(out)}")
    return out


def split_data(
    data: List[PaperReview],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.0,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[PaperReview], List[PaperReview], List[PaperReview]]:
    """
    Smart splitter:
      • If samples carry predefined split labels (PeerRead folder splits or
        the ICLR auto-split), use them directly. Dev counts as test — the
        pipeline only consumes train/test.
      • Otherwise fall back to a random shuffle using train_ratio/test_ratio.
    """
    train_has   = any(s.split == "train" for s in data)
    test_or_dev = any(s.split in ("dev", "test") for s in data)

    if train_has and test_or_dev:
        train = [s for s in data if s.split == "train"]
        test  = [s for s in data if s.split in ("dev", "test")]
        print(f"[split_data] using predefined splits: train={len(train)} test={len(test)} (dev folded into test)")
        return train, [], test

    # Fallback: random shuffle
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    n_train = int(len(data) * train_ratio)
    train = [data[i] for i in indices[:n_train]]
    test  = [data[i] for i in indices[n_train:]]
    print(f"[split_data] random split (no predef labels): train={len(train)} test={len(test)}")
    return train, [], test
