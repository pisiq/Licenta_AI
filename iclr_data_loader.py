"""
Robust data loader for ICLR paper review dataset.

Handles the folder structure:
data/
  ├── train/
  │   ├── parsed_pdfs/  (*.pdf.json files)
  │   ├── pdfs/         (raw PDFs - ignored)
  │   └── reviews/      (*.json files)
  ├── dev/
  │   ├── parsed_pdfs/
  │   ├── pdfs/
  │   └── reviews/
  └── test/
      ├── parsed_pdfs/
      ├── pdfs/
      └── reviews/
"""
import json
import os
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


class ICLRDataLoader:
    """
    Loads ICLR paper review data from folder structure.

    Handles:
    - Loading parsed PDF JSONs
    - Loading review JSONs
    - Matching papers with reviews by ID
    - Aggregating multiple reviews per paper
    - Validating data integrity
    """

    # Required score dimensions
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
        """
        Initialize data loader.

        Args:
            base_path: Path to data folder containing train/dev/test subdirectories
        """
        self.base_path = Path(base_path)

        if not self.base_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.base_path}")

        # Validate folder structure
        for split in ['train', 'dev', 'test']:
            split_path = self.base_path / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split folder does not exist: {split_path}")

            parsed_pdfs = split_path / "parsed_pdfs"
            reviews = split_path / "reviews"

            if not parsed_pdfs.exists():
                raise FileNotFoundError(f"parsed_pdfs folder does not exist: {parsed_pdfs}")
            if not reviews.exists():
                raise FileNotFoundError(f"reviews folder does not exist: {reviews}")

    def load_split(self, split: str, verbose: bool = True) -> List[PaperReview]:
        """
        Load data from a specific split.

        Args:
            split: One of 'train', 'dev', or 'test'
            verbose: Whether to print loading progress

        Returns:
            List of PaperReview objects
        """
        if split not in ['train', 'dev', 'test']:
            raise ValueError(f"Split must be one of ['train', 'dev', 'test'], got {split}")

        split_path = self.base_path / split
        parsed_pdfs_path = split_path / "parsed_pdfs"
        reviews_path = split_path / "reviews"

        if verbose:
            print(f"\n{'='*80}")
            print(f"Loading {split.upper()} split from {split_path}")
            print(f"{'='*80}")

        # Get all PDF files
        pdf_files = sorted(parsed_pdfs_path.glob("*.pdf.json"))

        if verbose:
            print(f"Found {len(pdf_files)} parsed PDF files")

        papers = []
        skipped = {'no_review': 0, 'invalid_scores': 0, 'empty_text': 0, 'parse_error': 0}

        for pdf_file in pdf_files:
            # Extract paper ID (e.g., "104" from "104.pdf.json")
            paper_id = pdf_file.stem.replace('.pdf', '')

            try:
                # Load parsed PDF
                paper_data = self._load_parsed_pdf(pdf_file)

                # Load review
                review_file = reviews_path / f"{paper_id}.json"
                if not review_file.exists():
                    skipped['no_review'] += 1
                    if verbose and len(papers) < 5:  # Only warn for first few
                        warnings.warn(f"No review found for paper {paper_id}")
                    continue

                review_data = self._load_review(review_file)

                # Extract and validate paper content
                title, abstract, full_text = self._extract_paper_content(paper_data, paper_id)

                if not full_text or len(full_text.strip()) < 100:
                    skipped['empty_text'] += 1
                    continue

                # Aggregate review scores
                scores = self._aggregate_review_scores(review_data, paper_id)

                if scores is None:
                    skipped['invalid_scores'] += 1
                    continue

                # Create PaperReview object (validates scores in __post_init__)
                paper = PaperReview(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    full_text=full_text,
                    scores=scores
                )

                papers.append(paper)

            except Exception as e:
                skipped['parse_error'] += 1
                if verbose and len(papers) < 5:
                    warnings.warn(f"Error loading paper {paper_id}: {e}")
                continue

        if verbose:
            print(f"\n✓ Successfully loaded {len(papers)} papers")
            if sum(skipped.values()) > 0:
                print(f"\n⚠ Skipped papers:")
                for reason, count in skipped.items():
                    if count > 0:
                        print(f"  - {reason}: {count}")

        return papers

    def _load_parsed_pdf(self, file_path: Path) -> Dict:
        """Load and parse a PDF JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_review(self, file_path: Path) -> Dict:
        """Load and parse a review JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_paper_content(self, paper_data: Dict, paper_id: str) -> Tuple[str, str, str]:
        """
        Extract title, abstract, and full text from parsed PDF.

        Args:
            paper_data: Parsed PDF JSON data
            paper_id: Paper ID for error messages

        Returns:
            (title, abstract, full_text)
        """
        # Extract title
        title = ""
        if 'metadata' in paper_data:
            metadata = paper_data['metadata']
            if isinstance(metadata, dict):
                title = metadata.get('title', '') or ""

        # If no title in metadata, use paper ID
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

        # Try multiple possible locations for content
        if 'metadata' in paper_data and isinstance(paper_data['metadata'], dict):
            sections = paper_data['metadata'].get('sections', [])
            if isinstance(sections, list):
                for section in sections:
                    if isinstance(section, dict) and 'text' in section:
                        full_text_parts.append(section['text'])

        # Fallback: concatenate all text fields
        if not full_text_parts:
            def extract_text_recursive(obj):
                """Recursively extract all text fields."""
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

        # Clean up text
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

        # Remove null bytes and control characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())

        return text.strip()

    def _aggregate_review_scores(self, review_data: Dict, paper_id: str) -> Optional[Dict[str, int]]:
        """
        Aggregate review scores from potentially multiple reviews.

        Args:
            review_data: Review JSON data
            paper_id: Paper ID for error messages

        Returns:
            Dictionary of {dimension: aggregated_score (1-5)} or None if invalid
        """
        # Extract reviews list
        reviews = review_data.get('reviews', [])
        if not isinstance(reviews, list) or len(reviews) == 0:
            return None

        # Collect scores per dimension
        dimension_scores = {dim: [] for dim in self.SCORE_DIMENSIONS}

        for review in reviews:
            if not isinstance(review, dict):
                continue

            for dim in self.SCORE_DIMENSIONS:
                # Try to find score for this dimension
                score = None

                # Try exact match
                if dim in review:
                    score = review[dim]
                # Try lowercase
                elif dim.lower() in review:
                    score = review[dim.lower()]

                # Convert to integer
                if score is not None:
                    try:
                        score_int = int(float(str(score).strip()))
                        if 1 <= score_int <= 5:
                            dimension_scores[dim].append(score_int)
                    except (ValueError, TypeError):
                        pass

        # Aggregate scores (mean and round)
        aggregated = {}
        for dim, scores_list in dimension_scores.items():
            if not scores_list:
                # If no valid scores for a dimension, skip this paper
                return None

            # Compute mean and round to nearest integer
            mean_score = np.mean(scores_list)
            aggregated_score = int(np.round(mean_score))

            # Ensure in valid range
            aggregated_score = np.clip(aggregated_score, 1, 5)
            aggregated[dim] = aggregated_score

        return aggregated

    def load_all_splits(self, verbose: bool = True) -> Tuple[List[PaperReview], List[PaperReview], List[PaperReview]]:
        """
        Load all splits (train, dev, test).

        Args:
            verbose: Whether to print loading progress

        Returns:
            (train_papers, dev_papers, test_papers)
        """
        train = self.load_split('train', verbose=verbose)
        dev = self.load_split('dev', verbose=verbose)
        test = self.load_split('test', verbose=verbose)

        if verbose:
            print(f"\n{'='*80}")
            print("DATA LOADING SUMMARY")
            print(f"{'='*80}")
            print(f"Train: {len(train)} papers")
            print(f"Dev:   {len(dev)} papers")
            print(f"Test:  {len(test)} papers")
            print(f"Total: {len(train) + len(dev) + len(test)} papers")
            print(f"{'='*80}\n")

        return train, dev, test

    def analyze_class_distribution(self, papers: List[PaperReview], split_name: str = ""):
        """
        Analyze and print class distribution for each score dimension.

        Args:
            papers: List of PaperReview objects
            split_name: Name of split for printing
        """
        print(f"\n{'='*80}")
        print(f"CLASS DISTRIBUTION ANALYSIS{' - ' + split_name.upper() if split_name else ''}")
        print(f"{'='*80}\n")

        for dim in self.SCORE_DIMENSIONS:
            scores = [paper.scores[dim] for paper in papers]
            counter = Counter(scores)

            print(f"{dim}:")
            for score in range(1, 6):
                count = counter.get(score, 0)
                pct = (count / len(scores) * 100) if scores else 0
                bar = '█' * int(pct / 2)  # Scale bar to fit
                print(f"  Score {score}: {count:4d} ({pct:5.1f}%) {bar}")

            # Check for severe imbalance
            max_count = max(counter.values()) if counter else 0
            min_count = min(counter.values()) if counter else 0
            if max_count > 0 and min_count > 0:
                ratio = max_count / min_count
                if ratio > 5:
                    print(f"  ⚠ WARNING: Severe class imbalance (ratio: {ratio:.1f}:1)")
            print()


class ICLRDataset(Dataset):
    """
    PyTorch Dataset for ICLR paper reviews.

    Tokenization is performed on-the-fly during batching (not pre-tokenized).
    """

    def __init__(
        self,
        papers: List[PaperReview],
        tokenizer,
        max_length: int = 4096,
        score_dimensions: List[str] = None
    ):
        """
        Initialize dataset.

        Args:
            papers: List of PaperReview objects
            tokenizer: Huggingface tokenizer
            max_length: Maximum sequence length
            score_dimensions: List of score dimensions to use
        """
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_dimensions = score_dimensions or ICLRDataLoader.SCORE_DIMENSIONS

        # Validate all papers have required dimensions
        for paper in self.papers:
            for dim in self.score_dimensions:
                if dim not in paper.scores:
                    raise ValueError(f"Paper {paper.paper_id} missing score for {dim}")

    def __len__(self) -> int:
        return len(self.papers)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item.

        Returns:
            Dictionary with:
                - input_ids: tokenized text
                - attention_mask: attention mask
                - labels: dict of {dimension: score (0-4)}
        """
        paper = self.papers[idx]

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
            labels[dim] = paper.scores[dim] - 1  # Convert to 0-indexed

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'paper_id': paper.paper_id  # For debugging
        }


# SANITY CHECK FUNCTIONS

def run_sanity_checks(
    train_papers: List[PaperReview],
    dev_papers: List[PaperReview],
    test_papers: List[PaperReview]
):
    """
    Run comprehensive sanity checks on loaded data.

    Args:
        train_papers: Training set papers
        dev_papers: Dev set papers
        test_papers: Test set papers
    """
    print(f"\n{'='*80}")
    print("SANITY CHECKS")
    print(f"{'='*80}\n")

    all_papers = train_papers + dev_papers + test_papers
    loader = ICLRDataLoader()

    # Check 1: Sample counts
    print("✓ Check 1: Sample counts")
    print(f"  Train: {len(train_papers)} samples")
    print(f"  Dev:   {len(dev_papers)} samples")
    print(f"  Test:  {len(test_papers)} samples")
    print(f"  Total: {len(all_papers)} samples")

    if len(all_papers) == 0:
        print("\n❌ FAILED: No samples loaded!\n")
        return False

    # Check 2: No labels outside 1-5
    print("\n✓ Check 2: Label validation")
    invalid_labels = 0
    for paper in all_papers:
        for dim, score in paper.scores.items():
            if not (1 <= score <= 5):
                invalid_labels += 1
                print(f"  ❌ Invalid score {score} for {dim} in paper {paper.paper_id}")

    if invalid_labels == 0:
        print("  ✓ All labels in valid range [1, 5]")
    else:
        print(f"\n❌ FAILED: {invalid_labels} invalid labels found!\n")
        return False

    # Check 3: No empty texts
    print("\n✓ Check 3: Text content validation")
    empty_texts = 0
    for paper in all_papers:
        if not paper.full_text or len(paper.full_text.strip()) < 100:
            empty_texts += 1
            print(f"  ❌ Empty/too short text in paper {paper.paper_id}")

    if empty_texts == 0:
        print("  ✓ All papers have valid text content")
    else:
        print(f"\n❌ FAILED: {empty_texts} papers with empty/invalid text!\n")
        return False

    # Check 4: Example sample
    print("\n✓ Check 4: Example sample")
    example = train_papers[0]
    print(f"  Paper ID: {example.paper_id}")
    print(f"  Title: {example.title[:80]}...")
    print(f"  Abstract length: {len(example.abstract)} chars")
    print(f"  Full text length: {len(example.full_text)} chars")
    print(f"  Scores:")
    for dim, score in example.scores.items():
        print(f"    {dim}: {score}")

    # Check 5: Class distribution analysis
    print("\n✓ Check 5: Class distribution")
    loader.analyze_class_distribution(train_papers, "TRAIN")

    print(f"\n{'='*80}")
    print("✅ ALL SANITY CHECKS PASSED!")
    print(f"{'='*80}\n")

    return True

