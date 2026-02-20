# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Test Installation
```powershell
python test_pipeline.py
```

## Option A: Run Complete Demo (10-15 minutes)

This runs the full pipeline end-to-end with sample data:

```powershell
python example.py
```

This will:
- Generate 100 sample papers with reviews
- Train a model for 3 epochs
- Evaluate on test set
- Make sample predictions
- Save the model

## Option B: Step-by-Step Training

### 1. Generate Sample Data
```powershell
python generate_sample_data.py
```

### 2. Train Model
```powershell
python train.py --data_path ./data/papers_reviews.json --output_dir ./outputs --num_epochs 5
```

### 3. Make Predictions

```python
from inference import load_model_for_inference

# Load model
predictor = load_model_for_inference("./outputs/best_model.pt")

# Predict
paper_text = """
Your paper text here...
"""

predictions = predictor.predict_scores(paper_text=paper_text)
print(predictions)
```

## Using Your Own Data

Create a JSON file with this structure:

```json
[
  {
    "title": "Your Paper Title",
    "abstract": "Your paper abstract...",
    "full_text": "Full paper text...",
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
      }
    ]
  }
]
```

Then train:

```powershell
python train.py --data_path ./your_data.json --output_dir ./outputs
```

## Key Configuration Options

Edit `config.py` to customize:

- `base_model_name`: Transformer model (default: `allenai/longformer-base-4096`)
- `max_length`: Max sequence length (default: 4096)
- `learning_rate`: Learning rate (default: 2e-5)
- `num_epochs`: Training epochs (default: 20)
- `train_batch_size`: Batch size (default: 4)

## Monitoring Training

View training progress in TensorBoard:

```powershell
tensorboard --logdir ./logs
```

Then open: http://localhost:6006

## Common Issues

**Out of Memory?**
- Reduce batch size: `--batch_size 2`
- Use smaller model: `--base_model distilbert-base-uncased`
- Reduce max_length in `config.py`

**Slow training?**
- Use GPU if available
- Reduce epochs for testing: `--num_epochs 3`
- Use smaller model for experimentation

**Need better performance?**
- Use SciBERT: `--base_model allenai/scibert_scivocab_uncased`
- Increase training epochs
- Get more training data
- Enable class weights (default: enabled)

## Next Steps

1. ✅ Run tests: `python test_pipeline.py`
2. ✅ Run demo: `python example.py`
3. ✅ Prepare your data
4. ✅ Train on real data
5. ✅ Evaluate and iterate
6. ✅ Deploy for inference

## Files Overview

- `train.py` - Main training script
- `inference.py` - Prediction utilities
- `model.py` - Model architecture
- `config.py` - Configuration
- `data_preprocessing.py` - Data utilities
- `metrics.py` - Evaluation metrics
- `trainer.py` - Training loop
- `example.py` - Complete demo
- `test_pipeline.py` - Unit tests

## Support

Check `README.md` for detailed documentation.

