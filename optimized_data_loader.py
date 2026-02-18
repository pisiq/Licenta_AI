"""
OPTIMIZED data loader for ICLR paper review dataset.

Key optimizations:
1. Lazy loading - only load papers when needed
2. Return raw text from __getitem__ (no tokenization)
3. Batched tokenization in collate_fn
4. Memory-efficient indexing
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from collections import Counter
import warnings


@dataclass
class PaperReview:
    """Data structure for a paper with aggregated review scores."""
    paper_id: str
    title: str
    abstract: str
    full_text: str
    scores: Dict[str, int]  # dimension -> score (1-5)

    def __post_init__(self):
        """Validate scores are in valid range."""
        for dim, score in self.scores.items():
            if not (1 <= score <= 5):
                raise ValueError(f"Score {score} for {dim} in paper {self.paper_id} is out of range [1,5]")


class OptimizedICLRDataLoader:
    """
    Memory-efficient ICLR data loader with lazy loading.
    """

    SCORE_DIMENSIONS = [
        "IMPACT",
        "SUBSTANCE",
        "APPROPRIATENESS",
        "MEANINGFUL_COMPARISON",
        "SOUNDNESS_CORRECTNESS",
        "ORIGINALITY",
        "CLARITY",
        "RECOMMENDATION"
    ]

    def __init__(self, base_path: str = "C:/Facultate/Licenta/data"):
        """Initialize data loader."""
        self.base_path = Path(base_path)

        if not self.base_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.base_path}")

        # Validate folder structure
        for split in ['train', 'dev', 'test']:
            split_path = self.base_path / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split folder does not exist: {split_path}")

    def get_paper_ids(self, split: str) -> List[str]:
        """
        Get list of paper IDs for a split without loading data.

        Args:
            split: One of 'train', 'dev', or 'test'

        Returns:
            List of paper IDs
        """
        if split not in ['train', 'dev', 'test']:
            raise ValueError(f"Split must be one of ['train', 'dev', 'test'], got {split}")

        split_path = self.base_path / split
        parsed_pdfs_path = split_path / "parsed_pdfs"
        reviews_path = split_path / "reviews"

        # Get all PDF files
        pdf_files = sorted(parsed_pdfs_path.glob("*.pdf.json"))

        # Filter to only papers with matching reviews
        paper_ids = []
        for pdf_file in pdf_files:
            paper_id = pdf_file.stem.replace('.pdf', '')
            review_file = reviews_path / f"{paper_id}.json"
            if review_file.exists():
                paper_ids.append(paper_id)

        return paper_ids

    def load_paper(self, split: str, paper_id: str) -> Optional[PaperReview]:
        """
        Load a single paper by ID (lazy loading).

        Args:
            split: One of 'train', 'dev', or 'test'
            paper_id: Paper ID

        Returns:
            PaperReview object or None if invalid
        """
        split_path = self.base_path / split
        parsed_pdfs_path = split_path / "parsed_pdfs"
        reviews_path = split_path / "reviews"

        pdf_file = parsed_pdfs_path / f"{paper_id}.pdf.json"
        review_file = reviews_path / f"{paper_id}.json"

        if not pdf_file.exists() or not review_file.exists():
            return None

        try:
            # Load parsed PDF
            with open(pdf_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)

            # Load review
            with open(review_file, 'r', encoding='utf-8') as f:
                review_data = json.load(f)

            # Extract content
            title, abstract, full_text = self._extract_paper_content(paper_data, paper_id)

            if not full_text or len(full_text.strip()) < 100:
                return None

            # Aggregate scores
            scores = self._aggregate_review_scores(review_data, paper_id)

            if scores is None:
                return None

            return PaperReview(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                full_text=full_text,
                scores=scores
            )

        except Exception:
            return None

    def _extract_paper_content(self, paper_data: Dict, paper_id: str) -> Tuple[str, str, str]:
        """Extract title, abstract, and full text from parsed PDF."""
        # Extract title
        title = ""
        if 'metadata' in paper_data:
            metadata = paper_data['metadata']
            if isinstance(metadata, dict):
                title = metadata.get('title', '') or ""

        if not title:
            title = f"Paper {paper_id}"

        # Extract abstract
        abstract = paper_data.get('abstract', '')
        if not abstract and 'metadata' in paper_data:
            metadata = paper_data.get('metadata', {})
            if isinstance(metadata, dict):
                abstract = metadata.get('abstractText', '') or ""

        # Extract full text from sections
        full_text_parts = []

        if 'metadata' in paper_data and isinstance(paper_data['metadata'], dict):
            sections = paper_data['metadata'].get('sections', [])
            if isinstance(sections, list):
                for section in sections:
                    if isinstance(section, dict) and 'text' in section:
                        full_text_parts.append(section['text'])

        # Fallback: extract all text recursively
        if not full_text_parts:
            def extract_text_recursive(obj):
                texts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'text' and isinstance(value, str):
                            texts.append(value)
                        elif isinstance(value, (dict, list)):
                            texts.extend(extract_text_recursive(value))
                elif isinstance(obj, list):
                    for item in obj:
                        texts.extend(extract_text_recursive(item))
                return texts

            full_text_parts = extract_text_recursive(paper_data)

        full_text = ' '.join(full_text_parts).strip()

        # Clean text
        title = self._clean_text(title)
        abstract = self._clean_text(abstract)
        full_text = self._clean_text(full_text)

        return title, abstract, full_text

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove control characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())

        return text.strip()

    def _aggregate_review_scores(self, review_data: Dict, paper_id: str) -> Optional[Dict[str, int]]:
        """Aggregate review scores from potentially multiple reviews."""
        reviews = review_data.get('reviews', [])
        if not isinstance(reviews, list) or len(reviews) == 0:
            return None

        # Collect scores per dimension
        dimension_scores = {dim: [] for dim in self.SCORE_DIMENSIONS}

        for review in reviews:
            if not isinstance(review, dict):
                continue

            for dim in self.SCORE_DIMENSIONS:
                score = None

                if dim in review:
                    score = review[dim]
                elif dim.lower() in review:
                    score = review[dim.lower()]

                if score is not None:
                    try:
                        score_int = int(float(str(score).strip()))
                        if 1 <= score_int <= 5:
                            dimension_scores[dim].append(score_int)
                    except (ValueError, TypeError):
                        pass

        # Aggregate scores
        aggregated = {}
        for dim, scores_list in dimension_scores.items():
            if not scores_list:
                return None

            mean_score = np.mean(scores_list)
            aggregated_score = int(np.round(mean_score))
            aggregated_score = np.clip(aggregated_score, 1, 5)
            aggregated[dim] = aggregated_score

        return aggregated


class OptimizedICLRDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset with lazy loading.

    Key optimizations:
    - Returns RAW TEXT only (no tokenization in __getitem__)
    - Lazy loads papers on demand
    - Minimal memory footprint
    """

    def __init__(
        self,
        data_loader: OptimizedICLRDataLoader,
        split: str,
        paper_ids: List[str],
        score_dimensions: List[str] = None
    ):
        """
        Initialize dataset.

        Args:
            data_loader: OptimizedICLRDataLoader instance
            split: Split name ('train', 'dev', or 'test')
            paper_ids: List of paper IDs to include
            score_dimensions: List of score dimensions to use
        """
        self.data_loader = data_loader
        self.split = split
        self.paper_ids = paper_ids
        self.score_dimensions = score_dimensions or OptimizedICLRDataLoader.SCORE_DIMENSIONS

        # Cache for loaded papers (optional - can be disabled for even lower memory)
        self._cache = {}

    def __len__(self) -> int:
        return len(self.paper_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item - returns RAW TEXT only.

        Returns:
            Dictionary with:
                - text: combined text (title + abstract + full_text)
                - labels: dict of {dimension: score (0-4)}
                - paper_id: paper ID (for debugging)
        """
        paper_id = self.paper_ids[idx]

        # Check cache first
        if paper_id in self._cache:
            paper = self._cache[paper_id]
        else:
            # Lazy load
            paper = self.data_loader.load_paper(self.split, paper_id)
            if paper is None:
                # Fallback for invalid papers
                return {
                    'text': "",
                    'labels': {dim: 2 for dim in self.score_dimensions},  # Default middle score
                    'paper_id': paper_id
                }
            # Cache (optional)
            self._cache[paper_id] = paper

        # Combine text
        combined_text = f"{paper.title}\n\n{paper.abstract}\n\n{paper.full_text}"

        # Extract labels (convert from 1-5 to 0-4)
        labels = {dim: paper.scores[dim] - 1 for dim in self.score_dimensions}

        return {
            'text': combined_text,
            'labels': labels,
            'paper_id': paper_id
        }


def optimized_collate_fn(batch, tokenizer, max_length=4096, device='cuda'):
    """
    Optimized collate function with batched tokenization.

    Args:
        batch: List of samples from dataset
        tokenizer: Huggingface tokenizer
        max_length: Maximum sequence length
        device: Target device for tensors

    Returns:
        Batched and tokenized data
    """
    # Extract texts and labels
    texts = [item['text'] for item in batch]
    paper_ids = [item['paper_id'] for item in batch]

    # Batched tokenization (much faster than individual)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Collect labels
    labels = {}
    score_dimensions = batch[0]['labels'].keys()
    for dim in score_dimensions:
        labels[dim] = torch.tensor([item['labels'][dim] for item in batch])

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels,
        'paper_ids': paper_ids
    }

