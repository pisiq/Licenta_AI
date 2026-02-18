# Multi-Task Ordinal Classification for Scientific Paper Review Scoring

A robust machine learning pipeline for automatically predicting review scores for scientific papers across multiple dimensions using ordinal classification.

## Overview

This system takes the full text of a scientific paper and predicts review scores (1-5) for eight dimensions:

- **IMPACT** - Expected impact on the field
- **SUBSTANCE** - Quality and depth of content
- **APPROPRIATENESS** - Fit for the venue
- **MEANINGFUL_COMPARISON** - Quality of experimental comparisons
- **SOUNDNESS_CORRECTNESS** - Technical correctness
- **ORIGINALITY** - Novelty of contributions
- **CLARITY** - Writing quality and presentation
- **RECOMMENDATION** - Overall recommendation

**Key Features:**
- ✅ Ordinal classification (NOT regression)
- ✅ Multi-task learning with shared encoder
- ✅ Transformer-based text encoding (Longformer for long documents)
- ✅ Proper handling of class imbalance
- ✅ Comprehensive metrics (QWK, F1, accuracy, Spearman correlation)
- ✅ Early stopping based on dev set QWK
- ✅ Clean preprocessing pipeline
- ✅ Modular and extensible design

## Project Structure

```
Licenta/
├── config.py                    # Configuration classes
├── data_preprocessing.py        # Data loading and preprocessing
├── model.py                     # Multi-task ordinal classifier
├── metrics.py                   # Evaluation metrics (QWK, F1, etc.)
├── trainer.py                   # Training loop and utilities
├── train.py                     # Main training script
├── inference.py                 # Prediction interface
├── generate_sample_data.py      # Sample data generator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### 1. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data (for testing)

```powershell
python generate_sample_data.py
```

This creates `./data/papers_reviews.json` with synthetic papers and reviews.

### 2. Train the Model

```powershell
python train.py --data_path ./data/papers_reviews.json --output_dir ./outputs --num_epochs 5
```

**Training Arguments:**
- `--data_path`: Path to training data JSON
- `--output_dir`: Directory to save model checkpoints
- `--base_model`: Base transformer model (default: `allenai/longformer-base-4096`)
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of training epochs

### 3. Make Predictions

```python
from inference import load_model_for_inference

# Load trained model
predictor = load_model_for_inference(
    checkpoint_path="./outputs/best_model.pt"
)

# Predict scores for a paper
paper_text = "Your full paper text here..."
predictions = predictor.predict_scores(
    paper_text=paper_text,
    title="Paper Title",
    abstract="Paper abstract"
)

print(predictions)
# Output: {'IMPACT': 4, 'SUBSTANCE': 5, 'CLARITY': 3, ...}

# Get probability distributions
probs = predictor.predict_probabilities(
    paper_text=paper_text,
    title="Paper Title",
    abstract="Paper abstract"
)

# Get predictions with confidence
results = predictor.predict_with_confidence(
    paper_text=paper_text,
    title="Paper Title",
    abstract="Paper abstract"
)
```

## Data Format

The training data should be a JSON file with the following structure:

```json
[
  {
    "title": "Paper Title",
    "abstract": "Paper abstract text...",
    "full_text": "Full paper text including all sections...",
    "reviews": [
      {
        "IMPACT": 4,
        "SUBSTANCE": 5,
        "APPROPRIATENESS": 4,
        "MEANINGFUL_COMPARISON": 3,
        "SOUNDNESS_CORRECTNESS": 4,
        "ORIGINALITY": 5,
        "CLARITY": 4,
        "RECOMMENDATION": 4
      },
      {
        "IMPACT": 3,
        "SUBSTANCE": 4,
        ...
      }
    ]
  },
  ...
]
```

**Notes:**
- Each paper can have multiple reviews
- Scores are aggregated by computing mean and rounding to nearest integer
- All scores must be integers from 1 to 5

## Model Architecture

### Shared Encoder
- Uses a pre-trained transformer (default: Longformer for 4096 tokens)
- Handles long scientific papers efficiently
- Optional hierarchical encoding for very long documents

### Multi-Task Heads
- Separate classification head for each score dimension
- Each head outputs logits for 5 classes (scores 1-5)
- Softmax activation for probability distributions

### Loss Function
- Cross-entropy loss for each dimension
- Optional class weights for handling imbalance
- Total loss = average across all dimensions

## Training Details

### Optimization
- **Optimizer:** AdamW with weight decay
- **Learning rate:** 2e-5 with linear warmup
- **Gradient clipping:** Max norm 1.0
- **Batch size:** 4 (configurable)
- **Gradient accumulation:** 4 steps

### Early Stopping
- Monitors average QWK across all dimensions on dev set
- Patience: 3 epochs without improvement
- Best model is automatically saved

### Metrics

For each dimension:
- **Accuracy** - Exact match accuracy
- **Macro F1** - F1 score averaged across classes
- **Quadratic Weighted Kappa (QWK)** - Primary metric for ordinal classification
- **Spearman Correlation** - Rank correlation
- **MAE** - Mean absolute error

Reported as:
- Per-dimension metrics
- Macro average across all dimensions

## Configuration

Edit `config.py` to customize:

### ModelConfig
```python
base_model_name = "allenai/longformer-base-4096"
max_length = 4096
num_classes = 5
```

### TrainingConfig
```python
learning_rate = 2e-5
num_epochs = 20
train_batch_size = 4
early_stopping_patience = 3
```

### DataConfig
```python
train_split = 0.8
dev_split = 0.1
test_split = 0.1
max_paper_length = 10000
```

## Experiment Tracking

Training logs are saved to TensorBoard:

```powershell
tensorboard --logdir ./logs
```

Metrics tracked:
- Training loss
- Dev set metrics (accuracy, F1, QWK, etc.)
- Per-dimension performance
- Learning rate schedule

## Advanced Usage

### Using Different Base Models

```powershell
# Use BERT
python train.py --base_model bert-base-uncased --data_path ./data/papers_reviews.json

# Use RoBERTa
python train.py --base_model roberta-base --data_path ./data/papers_reviews.json

# Use SciBERT (recommended for scientific papers)
python train.py --base_model allenai/scibert_scivocab_uncased --data_path ./data/papers_reviews.json
```

### Batch Inference

```python
from inference import load_model_for_inference, batch_predict

predictor = load_model_for_inference("./outputs/best_model.pt")

papers = [
    {
        "title": "Paper 1",
        "abstract": "Abstract 1",
        "full_text": "Full text 1..."
    },
    # ... more papers
]

results = batch_predict(predictor, papers, output_path="./predictions.json")
```

## Design Principles

1. **Ordinal Classification** - Treats scores as ordered categories, not continuous values
2. **Multi-Task Learning** - Shared representations improve generalization
3. **Robust Preprocessing** - Handles PDF artifacts, normalizes text, removes noise
4. **Proper Evaluation** - Uses QWK as primary metric for ordinal data
5. **Modular Design** - Easy to extend and customize
6. **Production Ready** - Clean inference API with confidence scores

## Citation

If you use this code in your research, please cite:

```bibtex
@software{paper_review_scorer,
  title={Multi-Task Ordinal Classification for Scientific Paper Review Scoring},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/paper-review-scorer}
}
```

## License

MIT License

## Troubleshooting

### Out of Memory
- Reduce `train_batch_size` in config
- Reduce `max_length` for shorter sequences
- Enable gradient accumulation

### Poor Performance
- Ensure data quality (check for missing/invalid labels)
- Try different base models (SciBERT for scientific papers)
- Increase training epochs
- Adjust learning rate
- Check class distribution and enable class weights

### Long Training Time
- Use GPU if available
- Reduce model size (use smaller base model)
- Reduce max sequence length
- Use fewer training epochs with early stopping

## Contact

For questions or issues, please open an issue on GitHub.

