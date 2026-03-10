"""
Unit tests for the pipeline components.
"""
import unittest
import torch
import numpy as np
from transformers import AutoTokenizer

from config import ModelConfig
from model import MultiTaskOrdinalClassifier
from metrics import quadratic_weighted_kappa, compute_metrics, compute_multi_task_metrics
from data_preprocessing import TextPreprocessor, ReviewAggregator, PaperReview


class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing."""

    def setUp(self):
        self.preprocessor = TextPreprocessor(
            normalize_whitespace=True,
            remove_references=True,
            max_length=1000,
            min_length=10
        )

    def test_clean_text(self):
        """Test basic text cleaning."""
        text = "This  is   a    test.\n\n\nWith   spaces."
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "a" * 2000
        truncated = self.preprocessor.truncate_text(long_text)
        self.assertEqual(len(truncated), 1000)

    def test_remove_references(self):
        """Test references removal."""
        text = "Introduction text.\n\nREFERENCES\n\n[1] Some reference"
        processed = self.preprocessor.remove_references_section(text)
        self.assertNotIn("Some reference", processed)
        self.assertIn("Introduction", processed)


class TestReviewAggregator(unittest.TestCase):
    """Test review score aggregation."""

    def setUp(self):
        self.aggregator = ReviewAggregator(method="mean_round", min_val=1, max_val=5)

    def test_mean_round_aggregation(self):
        """Test mean-round aggregation."""
        reviews = [
            {"IMPACT": 3, "CLARITY": 4},
            {"IMPACT": 4, "CLARITY": 5}
        ]
        aggregated = self.aggregator.aggregate_scores(reviews)
        self.assertEqual(aggregated["IMPACT"], 4)  # Round(3.5) = 4
        self.assertEqual(aggregated["CLARITY"], 4)  # Round(4.5) = 4

    def test_single_review(self):
        """Test single review (no aggregation needed)."""
        reviews = [{"IMPACT": 3, "CLARITY": 4}]
        aggregated = self.aggregator.aggregate_scores(reviews)
        self.assertEqual(aggregated["IMPACT"], 3)
        self.assertEqual(aggregated["CLARITY"], 4)

    def test_clipping(self):
        """Test score clipping to valid range."""
        reviews = [
            {"IMPACT": 5},
            {"IMPACT": 5}
        ]
        aggregated = self.aggregator.aggregate_scores(reviews)
        self.assertEqual(aggregated["IMPACT"], 5)
        self.assertLessEqual(aggregated["IMPACT"], 5)


class TestMetrics(unittest.TestCase):
    """Test metric computation."""

    def test_qwk_perfect(self):
        """Test QWK with perfect agreement."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        self.assertAlmostEqual(qwk, 1.0, places=5)

    def test_qwk_no_agreement(self):
        """Test QWK with no agreement."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([4, 4, 4, 4, 4])
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        self.assertLess(qwk, 0.5)

    def test_compute_metrics(self):
        """Test comprehensive metrics computation."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 3, 0, 2, 2])

        metrics = compute_metrics(y_true, y_pred)

        self.assertIn('accuracy', metrics)
        self.assertIn('macro_f1', metrics)
        self.assertIn('qwk', metrics)
        self.assertIn('spearman', metrics)

        # Check value ranges
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['qwk'], -1.0)
        self.assertLessEqual(metrics['qwk'], 1.0)

    def test_multi_task_metrics(self):
        """Test multi-task metrics computation."""
        predictions = {
            'DIM1': np.array([0, 1, 2, 3, 4]),
            'DIM2': np.array([1, 2, 3, 4, 4])
        }
        labels = {
            'DIM1': np.array([0, 1, 2, 3, 4]),
            'DIM2': np.array([1, 2, 3, 3, 4])
        }

        metrics = compute_multi_task_metrics(
            predictions, labels, ['DIM1', 'DIM2']
        )

        self.assertIn('per_dimension', metrics)
        self.assertIn('macro_avg', metrics)
        self.assertIn('avg_qwk', metrics)

        self.assertIn('DIM1', metrics['per_dimension'])
        self.assertIn('DIM2', metrics['per_dimension'])


class TestModel(unittest.TestCase):
    """Test model architecture."""

    def setUp(self):
        self.model_config = ModelConfig()
        self.model_config.base_model_name = "distilbert-base-uncased"
        self.model_config.max_length = 128

        self.model = MultiTaskOrdinalClassifier(
            base_model_name=self.model_config.base_model_name,
            score_dimensions=self.model_config.score_dimensions,
            num_classes=self.model_config.num_classes,
            dropout=0.1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name
        )

    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsNotNone(self.model.encoder)
        self.assertEqual(len(self.model.heads), len(self.model_config.score_dimensions))

        for dim in self.model_config.score_dimensions:
            self.assertIn(dim, self.model.heads)

    def test_forward_pass(self):
        """Test forward pass."""
        text = "This is a test paper about machine learning."
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        outputs = self.model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )

        self.assertIn('logits', outputs)
        self.assertEqual(len(outputs['logits']), len(self.model_config.score_dimensions))

        for dim, logits in outputs['logits'].items():
            self.assertEqual(logits.shape, (1, 5))  # batch_size=1, num_classes=5

    def test_forward_with_labels(self):
        """Test forward pass with labels (loss computation)."""
        text = "This is a test paper."
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = {dim: torch.tensor([2]) for dim in self.model_config.score_dimensions}

        outputs = self.model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            labels=labels
        )

        self.assertIn('loss', outputs)
        self.assertIn('per_task_loss', outputs)
        self.assertIsInstance(outputs['loss'].item(), float)

    def test_predict_scores(self):
        """Test score prediction."""
        text = "This is a test paper."
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        predictions = self.model.predict_scores(
            encoding['input_ids'],
            encoding['attention_mask']
        )

        self.assertEqual(len(predictions), len(self.model_config.score_dimensions))

        for dim, score in predictions.items():
            self.assertGreaterEqual(score, 1)
            self.assertLessEqual(score, 5)
            self.assertIsInstance(score, int)

    def test_predict_probabilities(self):
        """Test probability prediction."""
        text = "This is a test paper."
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        probabilities = self.model.predict_probabilities(
            encoding['input_ids'],
            encoding['attention_mask']
        )

        self.assertEqual(len(probabilities), len(self.model_config.score_dimensions))

        for dim, probs in probabilities.items():
            self.assertEqual(len(probs), 5)
            # Check probabilities sum to 1
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
            # Check all probabilities are valid
            self.assertTrue(torch.all(probs >= 0))
            self.assertTrue(torch.all(probs <= 1))


class TestDataStructures(unittest.TestCase):
    """Test data structures."""

    def test_paper_review_creation(self):
        """Test PaperReview creation."""
        paper = PaperReview(
            title="Test Title",
            abstract="Test Abstract",
            full_text="Test full text content.",
            scores={"IMPACT": 4, "CLARITY": 3}
        )

        self.assertEqual(paper.title, "Test Title")
        self.assertEqual(paper.scores["IMPACT"], 4)
        self.assertEqual(paper.scores["CLARITY"], 3)


def run_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestReviewAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDataStructures))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

