"""
Data preprocessing utilities for scientific paper review scoring.
"""
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class PaperReview:
    """Data structure for a paper with review scores."""
    title: str
    abstract: str
    full_text: str
    scores: Dict[str, int]  # dimension -> score (1-5)


class TextPreprocessor:
    """Cleans and normalizes scientific paper text."""

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
        """Clean PDF artifacts and normalize text."""
        if not text:
            return ""

        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null bytes
        text = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]', '', text)  # Control chars

        # Remove excessive unicode characters
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        # Remove or clean common PDF extraction issues
        text = re.sub(r'- ', '', text)  # Hyphenation
        text = re.sub(r'\n{3,}', '\n\n', text)  # Excessive newlines

        return text

    def remove_references_section(self, text: str) -> str:
        """Remove references section (often very long and less informative)."""
        if not self.remove_references:
            return text

        # Try to find references section
        patterns = [
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*References\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*Bibliography\s*\n',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Keep everything before references
                text = text[:match.start()]
                break

        return text

    def truncate_text(self, text: str) -> str:
        """Truncate text to maximum length."""
        if len(text) > self.max_length:
            text = text[:self.max_length]
        return text

    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.remove_references_section(text)
        text = self.truncate_text(text)
        return text


class ReviewAggregator:
    """Aggregates multiple reviews for the same paper."""

    def __init__(self, method: str = "mean_round", min_val: int = 1, max_val: int = 5):
        self.method = method
        self.min_val = min_val
        self.max_val = max_val

    def aggregate_scores(self, reviews: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Aggregate multiple review scores into a single score per dimension.

        Args:
            reviews: List of review dictionaries, each containing score dimensions

        Returns:
            Dictionary with aggregated scores (integers 1-5)
        """
        if len(reviews) == 1:
            return reviews[0]

        # Collect all scores per dimension
        dimension_scores = {}
        for review in reviews:
            for dim, score in review.items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)

        # Aggregate
        aggregated = {}
        for dim, scores in dimension_scores.items():
            if self.method == "mean_round":
                # Compute mean and round to nearest integer
                mean_score = np.mean(scores)
                aggregated_score = int(np.round(mean_score))
            elif self.method == "median":
                aggregated_score = int(np.median(scores))
            elif self.method == "mode":
                aggregated_score = int(np.bincount(scores).argmax())
            else:
                raise ValueError(f"Unknown aggregation method: {self.method}")

            # Clip to valid range
            aggregated_score = np.clip(aggregated_score, self.min_val, self.max_val)
            aggregated[dim] = aggregated_score

        return aggregated


class PaperReviewDataset(Dataset):
    """PyTorch Dataset for paper reviews."""

    def __init__(self,
                 data: List[PaperReview],
                 tokenizer,
                 max_length: int = 512,
                 score_dimensions: List[str] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_dimensions = score_dimensions or [
            "IMPACT", "SUBSTANCE", "APPROPRIATENESS", "MEANINGFUL_COMPARISON",
            "SOUNDNESS_CORRECTNESS", "ORIGINALITY", "CLARITY", "RECOMMENDATION"
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        paper = self.data[idx]

        # Combine title, abstract, and full text
        combined_text = f"{paper.title}\n\n{paper.abstract}\n\n{paper.full_text}"

        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract labels (convert from 1-5 to 0-4 for classification)
        labels = {}
        for dim in self.score_dimensions:
            if dim in paper.scores:
                labels[dim] = paper.scores[dim] - 1  # Convert to 0-indexed
            else:
                labels[dim] = -1  # Missing label indicator

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def load_and_preprocess_data(
    data_path: str,
    text_preprocessor: TextPreprocessor,
    review_aggregator: ReviewAggregator
) -> List[PaperReview]:
    """
    Load data from JSON file and preprocess.

    Expected JSON format:
    [
        {
            "title": "...",
            "abstract": "...",
            "full_text": "...",
            "reviews": [
                {"IMPACT": 4, "SUBSTANCE": 5, ...},
                {"IMPACT": 3, "SUBSTANCE": 4, ...}
            ]
        },
        ...
    ]
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []

    for item in raw_data:
        # Preprocess text
        title = text_preprocessor.clean_text(item.get('title', ''))
        abstract = text_preprocessor.clean_text(item.get('abstract', ''))
        full_text = text_preprocessor.preprocess(item.get('full_text', ''))

        # Skip if text is too short
        if len(full_text) < text_preprocessor.min_length:
            continue

        # Aggregate review scores
        reviews = item.get('reviews', [])
        if not reviews:
            continue

        aggregated_scores = review_aggregator.aggregate_scores(reviews)

        # Validate scores
        valid = True
        for score in aggregated_scores.values():
            if not (1 <= score <= 5):
                valid = False
                break

        if not valid:
            continue

        processed_data.append(PaperReview(
            title=title,
            abstract=abstract,
            full_text=full_text,
            scores=aggregated_scores
        ))

    return processed_data


def split_data(
    data: List[PaperReview],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[PaperReview], List[PaperReview], List[PaperReview]]:
    """Split data into train/dev/test sets."""
    np.random.seed(seed)

    # Shuffle data
    indices = np.random.permutation(len(data))

    # Calculate split points
    n_train = int(len(data) * train_ratio)
    n_dev = int(len(data) * dev_ratio)

    train_indices = indices[:n_train]
    dev_indices = indices[n_train:n_train + n_dev]
    test_indices = indices[n_train + n_dev:]

    train_data = [data[i] for i in train_indices]
    dev_data = [data[i] for i in dev_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, dev_data, test_data

