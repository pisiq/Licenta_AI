"""
Data preprocessing utilities for scientific paper review scoring.

Data layout expected on disk:
  data/<conference>/<split>/reviews/<ID>.json          <- review scores + comments
  data/<conference>/<split>/parsed_pdfs/<ID>.pdf.json  <- full paper text (may also be <ID>.json)

Every review file is matched to its parsed-PDF counterpart by ID.
Missing counterparts are silently skipped.
"""
import os
import re
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Score dimensions (matching PeerRead field names exactly)
# ---------------------------------------------------------------------------
SCORE_DIMENSIONS: List[str] = [
    "IMPACT",
    "SUBSTANCE",
    "APPROPRIATENESS",
    "MEANINGFUL_COMPARISON",
    "SOUNDNESS_CORRECTNESS",
    "ORIGINALITY",
    "CLARITY",
    "RECOMMENDATION",
]

# Conferences with ALL 8 score dimensions on a 1-5 scale (ACL, CoNLL)
PEERREAD_SCORED_CONFERENCES: List[str] = [
    "acl_2017",
    "conll_2016",
]

# ICLR 2017 has only RECOMMENDATION (1-10 scale) + REVIEWER_CONFIDENCE.
# It is normalised to 1-5 inside _extract_scores_and_comments and can be
# added to the conference list manually if more data is desired.
PEERREAD_ALL_CONFERENCES: List[str] = [
    "acl_2017",
    "conll_2016",
    "iclr_2017",
]

# Scale info per conference (max score value used for normalisation)
_CONFERENCE_SCORE_MAX: Dict[str, float] = {
    "acl_2017":   5.0,
    "conll_2016": 5.0,
    "iclr_2017":  10.0,  # ICLR uses 1-10 scale
}


# ===========================================================================
# Core data class
# ===========================================================================

@dataclass
class PaperReview:
    """Data structure for a paper paired with its review scores."""
    paper_id: str                              # e.g. "173"
    conference: str                            # e.g. "acl_2017"
    split: str                                 # "train" / "dev" / "test"
    title: str
    abstract: str
    paper_text: str                            # concatenated body sections from parsed PDF
    review_comments: str                       # concatenated reviewer comments
    combined_text: str                         # paper_text + [SEP] + review_comments
    scores: Dict[str, Optional[float]]         # dimension -> mean score (float), or None
    score_mask: Dict[str, bool]                # dimension -> True if score is valid

    @property
    def full_text(self) -> str:
        """Legacy alias so older code using .full_text still works."""
        return self.paper_text


# ===========================================================================
# Text preprocessing
# ===========================================================================

class TextPreprocessor:
    """Cleans and normalises scientific paper text."""

    def __init__(self,
                 normalize_whitespace: bool = True,
                 remove_references: bool = True,
                 max_length: int = 10000,
                 min_length: int = 100):
        self.normalize_whitespace = normalize_whitespace
        self.remove_references = remove_references
        self.max_length = max_length
        self.min_length = min_length

    def clean_text(self, text: str) -> str:
        """Clean PDF artefacts and normalise text."""
        if not text:
            return ""
        text = re.sub(r'\x00', '', text)
        text = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        text = re.sub(r'- ', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def remove_references_section(self, text: str) -> str:
        if not self.remove_references:
            return text
        for pattern in [r'\n\s*REFERENCES\s*\n', r'\n\s*References\s*\n',
                        r'\n\s*BIBLIOGRAPHY\s*\n', r'\n\s*Bibliography\s*\n']:
            m = re.search(pattern, text)
            if m:
                text = text[:m.start()]
                break
        return text

    def truncate_text(self, text: str) -> str:
        if len(text) > self.max_length:
            text = text[:self.max_length]
        return text

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        text = self.remove_references_section(text)
        text = self.truncate_text(text)
        return text


# ===========================================================================
# Review aggregator (kept for API compatibility - no longer used internally)
# ===========================================================================

class ReviewAggregator:
    """Aggregates multiple reviews for the same paper (legacy helper)."""

    def __init__(self, method: str = "mean_round", min_val: int = 1, max_val: int = 5):
        self.method = method
        self.min_val = min_val
        self.max_val = max_val

    def aggregate_scores(self, reviews: List[Dict[str, Any]]) -> Dict[str, int]:
        if len(reviews) == 1:
            return reviews[0]
        dimension_scores: Dict[str, List] = {}
        for review in reviews:
            for dim, score in review.items():
                dimension_scores.setdefault(dim, []).append(score)
        aggregated = {}
        for dim, scores in dimension_scores.items():
            if self.method == "mean_round":
                val = int(np.round(np.mean(scores)))
            elif self.method == "median":
                val = int(np.median(scores))
            else:
                val = int(np.round(np.mean(scores)))
            aggregated[dim] = int(np.clip(val, self.min_val, self.max_val))
        return aggregated


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class PaperReviewDataset(Dataset):
    """
    PyTorch Dataset for paper reviews.

    Training mode   (inference_mode=False):
        Input = TITLE + ABSTRACT + PAPER body + [SEP] + REVIEW comments
        The review text gives the model extra signal about why a paper
        received certain scores.

    Inference mode  (inference_mode=True):
        Input = TITLE + ABSTRACT + PAPER body  (no review)
        Mirrors real-world use: grading a brand-new paper before any
        reviewer has seen it.

    Labels / label_mask are always included so the same dataset object
    can be used for both training and evaluation.
    """

    def __init__(self,
                 data: List[PaperReview],
                 tokenizer,
                 max_length: int = 4096,
                 score_dimensions: List[str] = None,
                 print_summary: bool = True,
                 inference_mode: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_dimensions = score_dimensions or SCORE_DIMENSIONS
        self.inference_mode = inference_mode

        if print_summary and data:
            matched = len(data)
            mode_tag = "[INFERENCE - paper only]" if inference_mode else "[TRAINING - paper + review]"
            print(f"\n[OK] Successfully matched {matched} papers with their reviews. {mode_tag}")
            for dim in self.score_dimensions:
                valid = sum(1 for p in data if p.score_mask.get(dim, False))
                pct = 100 * valid // matched
                print(f"   {dim:<30}: {valid:>4}/{matched}  ({pct}%)")
            print()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch
        paper = self.data[idx]

        # ------------------------------------------------------------------
        # Inference mode : paper text only  (no review leakage)
        # Training mode  : paper text + reviewer comments
        # ------------------------------------------------------------------
        if self.inference_mode:
            input_text = _build_paper_only_text(paper.title, paper.abstract, paper.paper_text)
        else:
            input_text = paper.combined_text  # PAPER [SEP] REVIEW

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
            score = paper.scores.get(dim, None)
            valid = paper.score_mask.get(dim, False) and score is not None
            labels[dim] = torch.tensor(
                float(score) if valid else float("nan"), dtype=torch.float32
            )
            label_mask[dim] = torch.tensor(1.0 if valid else 0.0, dtype=torch.float32)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         labels,
            "label_mask":     label_mask,
        }


# ===========================================================================
# Private helpers for PeerRead loading
# ===========================================================================

def _find_parsed_pdf(parsed_pdfs_dir: str, paper_id: str) -> Optional[str]:
    """
    Return path to the parsed-PDF JSON for *paper_id*, or None if not found.
    Tries <ID>.pdf.json first (most conferences), then <ID>.json (some arxiv).
    """
    for candidate in (
        os.path.join(parsed_pdfs_dir, f"{paper_id}.pdf.json"),
        os.path.join(parsed_pdfs_dir, f"{paper_id}.json"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _extract_paper_text(
    pdf_data: dict, preprocessor: TextPreprocessor
) -> Tuple[str, str, str]:
    """
    Return (title, abstract, body_text) from a parsed-PDF JSON dict.

    PeerRead layout (confirmed):
        {"name": ...,
         "metadata": {
             "title": ...,
             "abstractText": ...,
             "sections": [{"heading": ..., "text": ...}, ...]
         }}
    Falls back gracefully when fields are missing.
    """
    metadata = pdf_data.get("metadata", pdf_data)  # arxiv dumps put fields at root

    title    = metadata.get("title", "") or ""
    abstract = (metadata.get("abstractText", "") or
                metadata.get("abstract", "") or "")

    # Build body from named sections (skip references)
    sections = metadata.get("sections") or pdf_data.get("sections") or []
    body_parts: List[str] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        heading = sec.get("heading") or ""
        text    = sec.get("text") or ""
        if re.search(r"\breferences?\b", heading, re.IGNORECASE):
            continue
        if heading:
            body_parts.append(f"### {heading}\n{text}")
        else:
            body_parts.append(text)

    body_text = "\n\n".join(body_parts)

    # Fallback to flat text field
    if not body_text:
        body_text = (metadata.get("text", "") or
                     pdf_data.get("text", "") or
                     pdf_data.get("full_text", "") or "")

    title     = preprocessor.clean_text(title)
    abstract  = preprocessor.clean_text(abstract)
    body_text = preprocessor.preprocess(body_text)

    return title, abstract, body_text


def _extract_scores_and_comments(
    review_data: dict,
    score_max: float = 5.0,
) -> Tuple[Dict[str, Optional[float]], Dict[str, bool], str]:
    """
    Return (scores, score_mask, all_comments) from a reviews JSON dict.

    Skips meta-reviews. Averages scores across valid reviewers.

    score_max : maximum value on this conference's scale (5 for ACL/CoNLL,
                10 for ICLR). Scores are normalised to [1, 5].
    """
    reviews_list = review_data.get("reviews", [])

    dim_values: Dict[str, List[float]] = {d: [] for d in SCORE_DIMENSIONS}
    comment_parts: List[str] = []

    for review in reviews_list:
        # Skip meta-reviews (both lowercase and uppercase key variants)
        is_meta = review.get("is_meta_review") or review.get("IS_META_REVIEW")
        if is_meta:
            continue

        # Collect reviewer comments
        comment = (review.get("comments", "") or review.get("comment", "") or "")
        if comment.strip():
            comment_parts.append(comment.strip())

        # Collect dimension scores
        for dim in SCORE_DIMENSIONS:
            raw = review.get(dim)
            if raw is None:
                continue
            try:
                val = float(raw)
                # Normalise to 1-5 scale if the conference uses a different range
                if score_max != 5.0:
                    # linear map: [1, score_max] -> [1, 5]
                    val = 1.0 + (val - 1.0) / (score_max - 1.0) * 4.0
                # Accept only values in the valid 1-5 range after normalisation
                if 1.0 <= val <= 5.0:
                    dim_values[dim].append(val)
            except (ValueError, TypeError):
                pass

    scores: Dict[str, Optional[float]] = {}
    score_mask: Dict[str, bool] = {}
    for dim in SCORE_DIMENSIONS:
        vals = dim_values[dim]
        if vals:
            scores[dim]     = float(np.mean(vals))
            score_mask[dim] = True
        else:
            scores[dim]     = None
            score_mask[dim] = False

    return scores, score_mask, "\n\n".join(comment_parts)


def _build_paper_only_text(
    title: str,
    abstract: str,
    body_text: str,
    max_paper_chars: int = 14_000,
) -> str:
    """
    Build the Longformer input for **inference** (no review available).
    Uses the full character budget for the paper itself.
    """
    return (
        f"TITLE: {title}\n\n"
        f"ABSTRACT: {abstract}\n\n"
        f"PAPER:\n{body_text[:max_paper_chars]}"
    )


def _build_combined_text(
    title: str,
    abstract: str,
    body_text: str,
    review_comments: str,
    max_paper_chars: int = 12_000,
    max_review_chars: int = 4_000,
) -> str:
    """
    Build the Longformer input for **training** (paper + reviewer comments).

    Format:
        TITLE: <title>

        ABSTRACT: <abstract>

        PAPER:
        <body_text (up to max_paper_chars)>

        [SEP]

        REVIEW:
        <review_comments (up to max_review_chars)>

    The [SEP] token acts as a clear boundary so the model can learn to
    distinguish paper content from reviewer language.
    """
    paper_part  = (
        f"TITLE: {title}\n\n"
        f"ABSTRACT: {abstract}\n\n"
        f"PAPER:\n{body_text[:max_paper_chars]}"
    )
    review_part = f"REVIEW:\n{review_comments[:max_review_chars]}"
    return f"{paper_part}\n\n[SEP]\n\n{review_part}"


# ===========================================================================
# Primary public loader
# ===========================================================================

def load_peerread_data(
    base_data_path: str,
    text_preprocessor: TextPreprocessor,
    conference_folders: List[str] = None,
    splits: List[str] = None,
    require_pdf: bool = True,
    min_body_length: int = 100,
    verbose: bool = True,
) -> List[PaperReview]:
    """
    Load PeerRead data by matching review files to parsed-PDF files via paper ID.

    Parameters
    ----------
    base_data_path      Root of the data directory (e.g. "./data").
    text_preprocessor   Used to clean / truncate extracted text.
    conference_folders  Sub-folders to scan. Defaults to PEERREAD_SCORED_CONFERENCES.
    splits              Which splits to load. Defaults to ["train", "dev", "test"].
    require_pdf         Skip samples whose parsed-PDF is missing (default True).
    min_body_length     Skip samples whose body text is shorter than this.
    verbose             Print a loading summary.

    Returns
    -------
    list[PaperReview]
    """
    if conference_folders is None:
        conference_folders = PEERREAD_SCORED_CONFERENCES
    if splits is None:
        splits = ["train", "dev", "test"]

    all_data: List[PaperReview] = []
    total_reviewed    = 0
    total_pdf_missing = 0
    total_no_scores   = 0
    total_too_short   = 0

    if verbose:
        confs_str = ", ".join(conference_folders)
        print(f"\nLoading PeerRead data from {len(conference_folders)} "
              f"conference(s): {confs_str}")

    for conf in conference_folders:
        conf_path = os.path.join(base_data_path, conf)
        if not os.path.isdir(conf_path):
            if verbose:
                print(f"  [SKIP] {conf} - folder not found")
            continue

        conf_count = 0

        for split in splits:
            reviews_dir     = os.path.join(conf_path, split, "reviews")
            parsed_pdfs_dir = os.path.join(conf_path, split, "parsed_pdfs")

            if not os.path.isdir(reviews_dir):
                continue

            review_files = sorted(glob.glob(os.path.join(reviews_dir, "*.json")))

            for review_path in review_files:
                total_reviewed += 1
                paper_id = os.path.basename(review_path).replace(".json", "")

                # -- 1. Locate parsed PDF ---------------------------------
                pdf_path = _find_parsed_pdf(parsed_pdfs_dir, paper_id)
                if pdf_path is None:
                    total_pdf_missing += 1
                    if require_pdf:
                        continue
                    pdf_data: dict = {}
                else:
                    try:
                        with open(pdf_path, encoding="utf-8") as f:
                            pdf_data = json.load(f)
                    except Exception:
                        total_pdf_missing += 1
                        if require_pdf:
                            continue
                        pdf_data = {}

                # -- 2. Extract paper text --------------------------------
                try:
                    title, abstract, body_text = _extract_paper_text(
                        pdf_data, text_preprocessor
                    )
                except Exception:
                    title, abstract, body_text = "", "", ""

                if require_pdf and len(body_text) < min_body_length:
                    total_too_short += 1
                    continue

                # -- 3. Load review scores + comments --------------------
                try:
                    with open(review_path, encoding="utf-8") as f:
                        review_data = json.load(f)
                except Exception:
                    continue

                scores, score_mask, review_comments = _extract_scores_and_comments(
                    review_data,
                    score_max=_CONFERENCE_SCORE_MAX.get(conf, 5.0),
                )

                if not any(score_mask.values()):
                    total_no_scores += 1
                    continue

                # -- 4. Build combined Longformer input -------------------
                combined_text = _build_combined_text(
                    title, abstract, body_text, review_comments
                )

                # -- 5. Append --------------------------------------------
                all_data.append(PaperReview(
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
                conf_count += 1

        if verbose:
            status = "OK" if conf_count > 0 else "WARNING - 0 papers"
            print(f"  [{status}] {conf}: {conf_count} papers")

    if verbose:
        bar = "-" * 60
        print(f"\n{bar}")
        print(f"  Review files found   : {total_reviewed}")
        print(f"  PDF missing / unread : {total_pdf_missing}")
        print(f"  No valid scores      : {total_no_scores}")
        print(f"  Body text too short  : {total_too_short}")
        print(f"  [OK] Successfully matched: {len(all_data)} papers")
        print(f"{bar}\n")

    return all_data


# ===========================================================================
# Backward-compatible wrappers (keep train.py unchanged)
# ===========================================================================

def load_all_peerread_data(
    base_data_path: str,
    text_preprocessor: TextPreprocessor,
    review_aggregator: ReviewAggregator = None,   # accepted but not used
    conference_folders: List[str] = None,
) -> List[PaperReview]:
    """Wrapper used by train.py --use_all_data."""
    return load_peerread_data(
        base_data_path    = base_data_path,
        text_preprocessor = text_preprocessor,
        conference_folders= conference_folders or PEERREAD_SCORED_CONFERENCES,
        verbose           = True,
    )


def load_and_preprocess_data(
    data_path: str,
    text_preprocessor: TextPreprocessor,
    review_aggregator: ReviewAggregator,
) -> List[PaperReview]:
    """
    Legacy loader: reads a single JSON file of the form
    [{"title": ..., "abstract": ..., "full_text": ..., "reviews": [...]}, ...]
    """
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

        scores: Dict[str, Optional[float]] = {}
        score_mask: Dict[str, bool] = {}
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
# Train / Dev / Test splitter
# ===========================================================================

def split_data(
    data: List[PaperReview],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PaperReview], List[PaperReview], List[PaperReview]]:
    """Randomly split data into train / dev / test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(data))

    n_train = int(len(data) * train_ratio)
    n_dev   = int(len(data) * dev_ratio)

    train_data = [data[i] for i in indices[:n_train]]
    dev_data   = [data[i] for i in indices[n_train:n_train + n_dev]]
    test_data  = [data[i] for i in indices[n_train + n_dev:]]

    return train_data, dev_data, test_data

