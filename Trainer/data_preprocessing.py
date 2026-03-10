"""
Data preprocessing utilities for scientific paper review scoring.

Supported conferences
---------------------
ACL 2017 / CoNLL 2016
    • Full 8-score suite on a 1-5 scale.
    • Pre-defined train/dev/test sub-folders from PeerRead.

ICLR 2017-2020
    • Only RECOMMENDATION (rating string, numeric 1-10 → normalised to 1-5)
      and REVIEWER_CONFIDENCE (confidence string, numeric 1-5).
    • Flat layout: <conf>/reviews/ and <conf>/parsed_pdfs/ with NO split folders.
    • Automatic 80/10/10 random split applied per ICLR year.

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

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels:     Dict[str, Any] = {}
        label_mask: Dict[str, Any] = {}
        for dim in self.score_dimensions:
            score = paper.scores.get(dim)
            valid = paper.score_mask.get(dim, False) and score is not None
            labels[dim]     = torch.tensor(float(score) if valid else float("nan"),
                                           dtype=torch.float32)
            label_mask[dim] = torch.tensor(1.0 if valid else 0.0,
                                           dtype=torch.float32)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         labels,
            "label_mask":     label_mask,
        }


# ===========================================================================
# Private helpers
# ===========================================================================

# ---- ICLR rating string → numeric value -----------------------------------

def _parse_iclr_rating(rating_str: Any) -> Optional[float]:
    """
    Extract the numeric part from ICLR rating strings.

    Examples:
        "8: Top 50% of accepted papers, clear accept"  → 8.0
        "10: Top 5% of accepted papers, seminal paper" → 10.0
        "3: Clear rejection"                           → 3.0
        8                                              → 8.0   (already numeric)
    """
    if rating_str is None:
        return None
    if isinstance(rating_str, (int, float)):
        val = float(rating_str)
        return val if 1.0 <= val <= 10.0 else None
    m = re.match(r'^\s*(\d+(?:\.\d+)?)', str(rating_str))
    if m:
        val = float(m.group(1))
        return val if 1.0 <= val <= 10.0 else None
    return None


def _parse_iclr_confidence(conf_str: Any) -> Optional[float]:
    """
    Extract numeric part from ICLR confidence strings.
    Scale is 1-5.
    """
    if conf_str is None:
        return None
    if isinstance(conf_str, (int, float)):
        val = float(conf_str)
        return val if 1.0 <= val <= 5.0 else None
    m = re.match(r'^\s*(\d+(?:\.\d+)?)', str(conf_str))
    if m:
        val = float(m.group(1))
        return val if 1.0 <= val <= 5.0 else None
    return None


def _normalise_to_1_5(val: float, raw_max: float) -> float:
    """Linear map: [1, raw_max] → [1, 5]."""
    if raw_max == 5.0:
        return val
    return 1.0 + (val - 1.0) / (raw_max - 1.0) * 4.0


# ---- Parsed-PDF lookup ----------------------------------------------------

def _find_parsed_pdf_acl_conll(parsed_pdfs_dir: str, paper_id: str) -> Optional[str]:
    """ACL/CoNLL: <ID>.pdf.json  or  <ID>.json"""
    for cand in (
        os.path.join(parsed_pdfs_dir, f"{paper_id}.pdf.json"),
        os.path.join(parsed_pdfs_dir, f"{paper_id}.json"),
    ):
        if os.path.exists(cand):
            return cand
    return None


def _find_parsed_pdf_iclr(parsed_pdfs_dir: str, paper_id: str) -> Optional[str]:
    """
    ICLR: review file is  ICLR_2017_<N>_review.json
          PDF file is      ICLR_2017_<N>_content.json
    paper_id is the full basename without extension, e.g. "ICLR_2017_42_review"
    We derive the content filename by replacing "_review" with "_content".
    """
    content_name = paper_id.replace("_review", "_content") + ".json"
    cand = os.path.join(parsed_pdfs_dir, content_name)
    if os.path.exists(cand):
        return cand
    # Fallback: try numeric ID only (e.g. "42") with both naming schemes
    numeric_match = re.search(r'_(\d+)_review$', paper_id)
    if numeric_match:
        n = numeric_match.group(1)
        for alt in (
            os.path.join(parsed_pdfs_dir, f"{n}_content.json"),
            os.path.join(parsed_pdfs_dir, f"{n}.json"),
        ):
            if os.path.exists(alt):
                return alt
    return None


# ---- Paper text extraction ------------------------------------------------

def _extract_paper_text(
    pdf_data: dict,
    preprocessor: TextPreprocessor,
) -> Tuple[str, str, str]:
    """Return (title, abstract, body_text) from a parsed-PDF JSON dict."""
    metadata = pdf_data.get("metadata", pdf_data)

    title    = metadata.get("title", "") or ""
    abstract = (metadata.get("abstractText", "")
                or metadata.get("abstract", "") or "")

    sections = metadata.get("sections") or pdf_data.get("sections") or []
    body_parts: List[str] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading") or ""
        text    = sec.get("text") or ""
        if re.search(r"\breferences?\b", heading, re.IGNORECASE):
            continue
        body_parts.append(f"### {heading}\n{text}" if heading else text)

    body_text = "\n\n".join(body_parts)
    if not body_text:
        body_text = (metadata.get("text", "") or pdf_data.get("text", "")
                     or pdf_data.get("full_text", "") or "")

    return (
        preprocessor.clean_text(title),
        preprocessor.clean_text(abstract),
        preprocessor.preprocess(body_text),
    )


# ---- Score + comment extraction -------------------------------------------

def _extract_scores_acl_conll(
    review_data: dict,
    score_max: float = 5.0,
) -> Tuple[Dict[str, Optional[float]], Dict[str, bool], str]:
    """
    Extract scores for ACL / CoNLL reviews.
    All SCORE_DIMENSIONS fields are plain numeric fields in each review dict.
    """
    reviews_list = review_data.get("reviews", [])

    dim_values: Dict[str, List[float]] = {d: [] for d in SCORE_DIMENSIONS}
    comment_parts: List[str] = []

    for review in reviews_list:
        is_meta = review.get("is_meta_review") or review.get("IS_META_REVIEW")
        if is_meta:
            continue

        comment = review.get("comments", "") or review.get("comment", "") or ""
        if comment.strip():
            comment_parts.append(comment.strip())

        for dim in SCORE_DIMENSIONS:
            raw = review.get(dim)
            if raw is None:
                continue
            try:
                val = float(raw)
                val = _normalise_to_1_5(val, score_max)
                if 1.0 <= val <= 5.0:
                    dim_values[dim].append(val)
            except (ValueError, TypeError):
                pass

    scores:     Dict[str, Optional[float]] = {}
    score_mask: Dict[str, bool]            = {}
    for dim in SCORE_DIMENSIONS:
        vals = dim_values[dim]
        if vals:
            scores[dim]     = float(np.mean(vals))
            score_mask[dim] = True
        else:
            scores[dim]     = None
            score_mask[dim] = False

    return scores, score_mask, "\n\n".join(comment_parts)


def _extract_scores_iclr(
    review_data: dict,
) -> Tuple[Dict[str, Optional[float]], Dict[str, bool], str]:
    """
    Extract scores for ICLR reviews.

    ICLR only has:
        rating      → RECOMMENDATION  (1-10 string → normalised to 1-5)
        confidence  → NOT included in SCORE_DIMENSIONS but used as comments

    All other 7 auxiliary dimensions are masked out (score_mask=False).
    """
    reviews_list = review_data.get("reviews", [])

    rating_vals: List[float] = []
    comment_parts: List[str] = []

    for review in reviews_list:
        # Skip meta-reviews
        is_meta = review.get("is_meta_review") or review.get("IS_META_REVIEW")
        if is_meta:
            continue

        # Collect reviewer text
        text = review.get("review", "") or review.get("comment", "") or ""
        if text.strip():
            comment_parts.append(text.strip())

        # Parse rating → RECOMMENDATION
        raw_rating = review.get("rating")
        val = _parse_iclr_rating(raw_rating)
        if val is not None:
            normalised = _normalise_to_1_5(val, 10.0)
            if 1.0 <= normalised <= 5.0:
                rating_vals.append(normalised)

    # Build scores / mask — only RECOMMENDATION is valid for ICLR
    scores:     Dict[str, Optional[float]] = {}
    score_mask: Dict[str, bool]            = {}
    for dim in SCORE_DIMENSIONS:
        if dim == "RECOMMENDATION" and rating_vals:
            scores[dim]     = float(np.mean(rating_vals))
            score_mask[dim] = True
        else:
            scores[dim]     = None
            score_mask[dim] = False

    return scores, score_mask, "\n\n".join(comment_parts)


# ---- Text builders --------------------------------------------------------

def _build_paper_only_text(
    title:     str,
    abstract:  str,
    body_text: str,
    max_paper_chars: int = 14_000,
) -> str:
    return (
        f"TITLE: {title}\n\n"
        f"ABSTRACT: {abstract}\n\n"
        f"PAPER:\n{body_text[:max_paper_chars]}"
    )


def _build_combined_text(
    title:           str,
    abstract:        str,
    body_text:       str,
    review_comments: str,
    max_paper_chars:  int = 12_000,
    max_review_chars: int = 4_000,
) -> str:
    paper_part  = (
        f"TITLE: {title}\n\n"
        f"ABSTRACT: {abstract}\n\n"
        f"PAPER:\n{body_text[:max_paper_chars]}"
    )
    review_part = f"REVIEW:\n{review_comments[:max_review_chars]}"
    return f"{paper_part}\n\n[SEP]\n\n{review_part}"


# ===========================================================================
# Loaders
# ===========================================================================

def _load_split_conference(
    conf:              str,
    conf_path:         str,
    splits:            List[str],
    preprocessor:      TextPreprocessor,
    require_pdf:       bool,
    min_body_length:   int,
    score_max:         float,
) -> Tuple[List[PaperReview], Dict[str, int]]:
    """Load a PeerRead conference that has pre-defined train/dev/test folders."""
    data: List[PaperReview] = []
    stats = dict(reviewed=0, pdf_missing=0, no_scores=0, too_short=0)

    for split in splits:
        reviews_dir     = os.path.join(conf_path, split, "reviews")
        parsed_pdfs_dir = os.path.join(conf_path, split, "parsed_pdfs")

        if not os.path.isdir(reviews_dir):
            continue

        for review_path in sorted(glob.glob(os.path.join(reviews_dir, "*.json"))):
            stats["reviewed"] += 1
            paper_id = os.path.basename(review_path).replace(".json", "")

            # Load parsed PDF
            pdf_path = _find_parsed_pdf_acl_conll(parsed_pdfs_dir, paper_id)
            pdf_data: dict = {}
            if pdf_path is None:
                stats["pdf_missing"] += 1
                if require_pdf:
                    continue
                pdf_data: dict = {}
            else:
                try:
                    with open(pdf_path, encoding="utf-8") as f:
                        pdf_data = json.load(f)
                except Exception:
                    stats["pdf_missing"] += 1
                    if require_pdf:
                        continue
                    pdf_data = {}

            # Extract paper text
            try:
                title, abstract, body_text = _extract_paper_text(pdf_data, preprocessor)
            except Exception:
                title, abstract, body_text = "", "", ""

            if require_pdf and len(body_text) < min_body_length:
                stats["too_short"] += 1
                continue

            # Load review
            try:
                with open(review_path, encoding="utf-8") as f:
                    review_data = json.load(f)
            except Exception:
                continue

            scores, score_mask, review_comments = _extract_scores_acl_conll(
                review_data, score_max
            )

            if not any(score_mask.values()):
                stats["no_scores"] += 1
                continue

            combined_text = _build_combined_text(
                title, abstract, body_text, review_comments
            )
            data.append(PaperReview(
                paper_id        = paper_id,
                conference      = conf,
                split           = split,
                title           = title,
                abstract        = abstract,
                paper_text      = body_text,
                review_comments = review_comments,
                combined_text   = combined_text,
                scores          = scores,
                score_mask      = score_mask,
            ))

    return data, stats


def _load_iclr_conference(
    conf:            str,
    conf_path:       str,
    preprocessor:    TextPreprocessor,
    require_pdf:     bool,
    min_body_length: int,
    train_ratio:     float = 0.8,
    dev_ratio:       float = 0.1,
    seed:            int   = 42,
) -> Tuple[List[PaperReview], Dict[str, int]]:
    """
    Load an ICLR conference (flat layout — no sub-split folders).
    Applies an automatic 80/10/10 random split.
    """
    reviews_dir     = os.path.join(conf_path, "reviews")
    parsed_pdfs_dir = os.path.join(conf_path, "parsed_pdfs")
    data: List[PaperReview] = []
    stats = dict(reviewed=0, pdf_missing=0, no_scores=0, too_short=0)

    if not os.path.isdir(reviews_dir):
        return data, stats

    for review_path in sorted(glob.glob(os.path.join(reviews_dir, "*.json"))):
        stats["reviewed"] += 1
        # paper_id = full stem, e.g. "ICLR_2017_42_review"
        paper_id = os.path.basename(review_path).replace(".json", "")

        # Load parsed PDF (uses _find_parsed_pdf_iclr)
        pdf_path = _find_parsed_pdf_iclr(parsed_pdfs_dir, paper_id)
        pdf_data: dict = {}
        if pdf_path is None:
            stats["pdf_missing"] += 1
            if require_pdf:
                continue
            pdf_data: dict = {}
        else:
            try:
                with open(pdf_path, encoding="utf-8") as f:
                    pdf_data = json.load(f)
            except Exception:
                stats["pdf_missing"] += 1
                if require_pdf:
                    continue
                pdf_data = {}

        try:
            title, abstract, body_text = _extract_paper_text(pdf_data, preprocessor)
        except Exception:
            title, abstract, body_text = "", "", ""

        if require_pdf and len(body_text) < min_body_length:
            stats["too_short"] += 1
            continue

        try:
            with open(review_path, encoding="utf-8") as f:
                review_data = json.load(f)
        except Exception:
            continue

        scores, score_mask, review_comments = _extract_scores_iclr(review_data)

        if not any(score_mask.values()):
            stats["no_scores"] += 1
            continue

        combined_text = _build_combined_text(
            title, abstract, body_text, review_comments
        )
        data.append(PaperReview(
            paper_id        = paper_id,
            conference      = conf,
            split           = "unassigned",   # will be reassigned below
            title           = title,
            abstract        = abstract,
            paper_text      = body_text,
            review_comments = review_comments,
            combined_text   = combined_text,
            scores          = scores,
            score_mask      = score_mask,
        ))

    # Auto-split this ICLR year
    rng = random.Random(seed)
    rng.shuffle(data)
    n       = len(data)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)
    for i, item in enumerate(data):
        if i < n_train:
            item.split = "train"
        elif i < n_train + n_dev:
            item.split = "dev"
        else:
            item.split = "test"

    return data, stats


# ===========================================================================
# Primary public loader
# ===========================================================================

def load_peerread_data(
    base_data_path:     str,
    text_preprocessor:  TextPreprocessor,
    conference_folders: List[str] = None,
    splits:             List[str] = None,
    require_pdf:        bool  = True,
    min_body_length:    int   = 100,
    verbose:            bool  = True,
    seed:               int   = 42,
) -> List[PaperReview]:
    """
    Load PeerRead data from ACL, CoNLL, and/or ICLR conferences.

    ACL / CoNLL : use pre-defined split folders.
    ICLR        : flat layout → automatic 80/10/10 split per year.
    """
    if conference_folders is None:
        conference_folders = PEERREAD_ALL_CONFERENCES
    if splits is None:
        splits = ["train", "dev", "test"]

    all_data: List[PaperReview] = []
    total_reviewed    = 0
    total_pdf_missing = 0
    total_no_scores   = 0
    total_too_short   = 0

    if verbose:
        print(f"\nLoading PeerRead data from {len(conference_folders)} conference(s): "
              f"{', '.join(conference_folders)}")

    for conf in conference_folders:
        conf_path = os.path.join(base_data_path, conf)
        if not os.path.isdir(conf_path):
            if verbose:
                print(f"  [SKIP] {conf} — folder not found at {conf_path}")
            continue

        is_iclr = conf.upper().startswith("ICLR")
        score_max = _CONFERENCE_SCORE_MAX.get(conf, 5.0)

        if is_iclr:
            conf_data, stats = _load_iclr_conference(
                conf, conf_path, text_preprocessor,
                require_pdf, min_body_length, seed=seed,
            )
        else:
            conf_data, stats = _load_split_conference(
                conf, conf_path, splits, text_preprocessor,
                require_pdf, min_body_length, score_max,
            )

        total_reviewed    += stats["reviewed"]
        total_pdf_missing += stats["pdf_missing"]
        total_no_scores   += stats["no_scores"]
        total_too_short   += stats["too_short"]

        # Only keep samples from requested splits
        conf_data = [s for s in conf_data if s.split in splits]
        all_data.extend(conf_data)

        if verbose:
            by_split = {}
            for s in conf_data:
                by_split[s.split] = by_split.get(s.split, 0) + 1
            split_str = "  ".join(f"{k}={v}" for k, v in sorted(by_split.items()))
            status = "OK" if conf_data else "WARNING — 0 papers loaded"
            print(f"  [{status}] {conf}: {len(conf_data)} papers  ({split_str})")
            # Score coverage per dimension
            if conf_data:
                for dim in SCORE_DIMENSIONS:
                    valid = sum(1 for p in conf_data if p.score_mask.get(dim, False))
                    pct   = 100 * valid // len(conf_data)
                    if pct > 0:
                        print(f"           {dim:<30}: {valid:>4}/{len(conf_data)}  ({pct}%)")

    if verbose:
        bar = "-" * 60
        print(f"\n{bar}")
        print(f"  Review files found   : {total_reviewed}")
        print(f"  PDF missing / unread : {total_pdf_missing}")
        print(f"  No valid scores      : {total_no_scores}")
        print(f"  Body text too short  : {total_too_short}")
        print(f"  [OK] Total matched   : {len(all_data)} papers")
        print(f"{bar}\n")

    return all_data


# ===========================================================================
# Backward-compatible wrappers
# ===========================================================================

def load_all_peerread_data(
    base_data_path:     str,
    text_preprocessor:  TextPreprocessor,
    review_aggregator:  ReviewAggregator = None,   # accepted but unused
    conference_folders: List[str] = None,
) -> List[PaperReview]:
    """Wrapper used by train.py --use_all_data."""
    return load_peerread_data(
        base_data_path     = base_data_path,
        text_preprocessor  = text_preprocessor,
        conference_folders = conference_folders or PEERREAD_ALL_CONFERENCES,
        verbose            = True,
    )


def load_and_preprocess_data(
    data_path:         str,
    text_preprocessor: TextPreprocessor,
    review_aggregator: ReviewAggregator,
) -> List[PaperReview]:
    """Legacy loader for a single flat JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed: List[PaperReview] = []
    for i, item in enumerate(raw_data):
        title     = text_preprocessor.clean_text(item.get("title", ""))
        abstract  = text_preprocessor.clean_text(item.get("abstract", ""))
        body_text = text_preprocessor.preprocess(item.get("full_text", ""))

        if len(body_text) < text_preprocessor.min_length:
            continue

        reviews = item.get("reviews", [])
        if not reviews:
            continue

        dim_values: Dict[str, List[float]] = {d: [] for d in SCORE_DIMENSIONS}
        for rev in reviews:
            for dim in SCORE_DIMENSIONS:
                raw = rev.get(dim)
                if raw is not None:
                    try:
                        val = float(raw)
                        if 1.0 <= val <= 5.0:
                            dim_values[dim].append(val)
                    except (ValueError, TypeError):
                        pass

        scores:     Dict[str, Optional[float]] = {}
        score_mask: Dict[str, bool]            = {}
        for dim in SCORE_DIMENSIONS:
            vals = dim_values[dim]
            if vals:
                scores[dim]     = float(np.mean(vals))
                score_mask[dim] = True
            else:
                scores[dim]     = None
                score_mask[dim] = False

        if not any(score_mask.values()):
            continue

        combined_text = _build_combined_text(title, abstract, body_text, "")
        processed.append(PaperReview(
            paper_id        = str(i),
            conference      = "legacy",
            split           = "train",
            title           = title,
            abstract        = abstract,
            paper_text      = body_text,
            review_comments = "",
            combined_text   = combined_text,
            scores          = scores,
            score_mask      = score_mask,
        ))
    return processed


# ===========================================================================
# Splitter  (used when caller wants a random split of already-loaded data)
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
    """
    has_predef = any(s.split in ("train", "dev", "test") for s in data)
    train_has  = any(s.split == "train" for s in data)
    dev_has    = any(s.split == "dev"   for s in data)
    test_has   = any(s.split == "test"  for s in data)

    if has_predef and train_has and dev_has and test_has:
        # Use pre-defined splits
        train = [s for s in data if s.split == "train"]
        dev   = [s for s in data if s.split == "dev"]
        test  = [s for s in data if s.split == "test"]
        return train, dev, test

    # Random fallback
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    n_train = int(len(data) * train_ratio)
    n_dev   = int(len(data) * dev_ratio)
    train   = [data[i] for i in indices[:n_train]]
    dev     = [data[i] for i in indices[n_train:n_train + n_dev]]
    test    = [data[i] for i in indices[n_train + n_dev:]]
    return train, dev, test
