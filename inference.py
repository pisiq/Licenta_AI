"""
Inference utilities for making predictions on new papers.
"""
import torch
from transformers import AutoTokenizer
from typing import Dict, List
import json

from model import MultiTaskOrdinalClassifier
from data_preprocessing import TextPreprocessor
from config import ModelConfig


class PaperReviewPredictor:
    """
    Predictor class for scoring scientific papers.

    Provides easy-to-use interface for making predictions on new papers.
    """

    def __init__(
        self,
        model: MultiTaskOrdinalClassifier,
        tokenizer,
        text_preprocessor: TextPreprocessor,
        device: torch.device = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.text_preprocessor = text_preprocessor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.model.eval()

    def predict_scores(
        self,
        paper_text: str,
        title: str = "",
        abstract: str = ""
    ) -> Dict[str, int]:
        """
        Predict review scores for a paper.

        Args:
            paper_text: Full text of the paper
            title: Paper title (optional)
            abstract: Paper abstract (optional)

        Returns:
            Dictionary of {dimension: predicted_score} (1-5)
        """
        # Preprocess text
        paper_text = self.text_preprocessor.preprocess(paper_text)
        title = self.text_preprocessor.clean_text(title)
        abstract = self.text_preprocessor.clean_text(abstract)

        # Combine text
        combined_text = f"{title}\n\n{abstract}\n\n{paper_text}"

        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        predictions = self.model.predict_scores(input_ids, attention_mask)

        return predictions

    def predict_probabilities(
        self,
        paper_text: str,
        title: str = "",
        abstract: str = ""
    ) -> Dict[str, List[float]]:
        """
        Predict class probability distributions for a paper.

        Args:
            paper_text: Full text of the paper
            title: Paper title (optional)
            abstract: Paper abstract (optional)

        Returns:
            Dictionary of {dimension: [prob_class_1, ..., prob_class_5]}
        """
        # Preprocess text
        paper_text = self.text_preprocessor.preprocess(paper_text)
        title = self.text_preprocessor.clean_text(title)
        abstract = self.text_preprocessor.clean_text(abstract)

        # Combine text
        combined_text = f"{title}\n\n{abstract}\n\n{paper_text}"

        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict probabilities
        probabilities = self.model.predict_probabilities(input_ids, attention_mask)

        # Convert to lists and format
        formatted_probs = {}
        for dim, probs in probabilities.items():
            # Map from 0-indexed classes to 1-5 scores
            formatted_probs[dim] = {
                f"score_{i+1}": float(probs[i]) for i in range(len(probs))
            }

        return formatted_probs

    def predict_with_confidence(
        self,
        paper_text: str,
        title: str = "",
        abstract: str = ""
    ) -> Dict[str, Dict]:
        """
        Predict scores with confidence scores.

        Args:
            paper_text: Full text of the paper
            title: Paper title (optional)
            abstract: Paper abstract (optional)

        Returns:
            Dictionary of {dimension: {score, confidence, probabilities}}
        """
        scores = self.predict_scores(paper_text, title, abstract)
        probabilities = self.predict_probabilities(paper_text, title, abstract)

        results = {}
        for dim in scores.keys():
            predicted_score = scores[dim]
            probs = probabilities[dim]

            # Confidence is the probability of the predicted class
            confidence = probs[f"score_{predicted_score}"]

            results[dim] = {
                'predicted_score': predicted_score,
                'confidence': confidence,
                'probabilities': probs
            }

        return results


def load_model_for_inference(
    checkpoint_path: str,
    model_config: ModelConfig = None,
    device: torch.device = None
) -> PaperReviewPredictor:
    """
    Load a trained model for inference.

    Args:
        checkpoint_path: Path to saved model checkpoint
        model_config: Model configuration (if None, uses default)
        device: Device to run inference on

    Returns:
        PaperReviewPredictor instance
    """
    if model_config is None:
        model_config = ModelConfig()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = MultiTaskOrdinalClassifier(
        base_model_name=model_config.base_model_name,
        score_dimensions=model_config.score_dimensions,
        num_classes=model_config.num_classes,
        dropout=model_config.hidden_dropout_prob
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)

    # Create text preprocessor
    text_preprocessor = TextPreprocessor(
        normalize_whitespace=True,
        remove_references=True,
        max_length=10000,
        min_length=100
    )

    # Create predictor
    predictor = PaperReviewPredictor(
        model=model,
        tokenizer=tokenizer,
        text_preprocessor=text_preprocessor,
        device=device
    )

    return predictor


def batch_predict(
    predictor: PaperReviewPredictor,
    papers: List[Dict[str, str]],
    output_path: str = None
) -> List[Dict]:
    """
    Make predictions on a batch of papers.

    Args:
        predictor: PaperReviewPredictor instance
        papers: List of dicts with 'title', 'abstract', 'full_text'
        output_path: Optional path to save results as JSON

    Returns:
        List of prediction results
    """
    results = []

    for i, paper in enumerate(papers):
        print(f"Processing paper {i+1}/{len(papers)}")

        predictions = predictor.predict_with_confidence(
            paper_text=paper.get('full_text', ''),
            title=paper.get('title', ''),
            abstract=paper.get('abstract', '')
        )

        result = {
            'title': paper.get('title', ''),
            'predictions': predictions
        }

        results.append(result)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results

