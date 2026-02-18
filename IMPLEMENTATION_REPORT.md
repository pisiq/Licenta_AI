# ICLR Paper Review Scorer - Complete Implementation Guide

## ✅ PHASE 1: SELF-AUDIT RESULTS

### Implementation Verified as CORRECT Ordinal Classification

#### ✓ 1. ORDINAL CLASSIFICATION (Not Regression)
- **Confirmed**: Uses classification heads with `nn.Linear(hidden_size, 5)`
- **Confirmed**: Applies softmax via `CrossEntropyLoss`
- **Confirmed**: Output dimension = 5 classes per head
- **Confirmed**: Labels: 1-5 (user-facing) → 0-4 (internal classification) → 1-5 (predictions)
- **NO regression heads exist**

#### ✓ 2. MULTI-TASK SETUP
- **Confirmed**: Shared transformer encoder (`self.encoder`)
- **Confirmed**: Separate classification heads in `nn.ModuleDict`
- **All 8 dimensions present:**
  1. IMPACT
  2. SUBSTANCE
  3. APPROPRIATENESS
  4. MEANINGFUL_COMPARISON
  5. SOUNDNESS_CORRECTNESS
  6. ORIGINALITY
  7. CLARITY
  8. RECOMMENDATION

#### ✓ 3. LOSS FUNCTION
- **Confirmed**: `nn.CrossEntropyLoss` used (model.py line 119)
- **Confirmed**: NO `MSELoss` anywhere in codebase
- **Confirmed**: Total loss = mean of per-dimension losses
- **Confirmed**: Optional class weights for imbalance

#### ✓ 4. METRICS IMPLEMENTED
- **Accuracy**: ✓ `accuracy_score()`
- **Macro F1**: ✓ `f1_score(average='macro')`
- **Quadratic Weighted Kappa (QWK)**: ✓ Primary metric
- **Spearman Correlation**: ✓ `spearmanr()`
- **MAE**: ✓ Mean Absolute Error (bonus)

#### ✓ 5. TRAINING LOGIC
- **Early Stopping**: ✓ Based on dev `avg_qwk` (macro average)
- **Gradient Clipping**: ✓ `max_grad_norm=1.0`
- **AdamW Optimizer**: ✓ With weight decay
- **Learning Rate Schedule**: ✓ Linear warmup + decay

---

## ✅ PHASE 2: ROBUST DATA LOADING IMPLEMENTED

### New Files Created

1. **`iclr_data_loader.py`** - Complete data loading system
   - `ICLRDataLoader` class
   - `ICLRDataset` class (PyTorch Dataset)
   - `PaperReview` dataclass
   - Sanity check functions

2. **`train_iclr.py`** - Updated training script
   - Uses robust data loader
   - Runs sanity checks before training
   - Comprehensive logging

3. **`quick_test.py`** - Quick verification script

### Data Loader Features

#### ✅ Robust Loading
- Loads from folder structure: `data/{train,dev,test}/{parsed_pdfs,reviews}/`
- Matches papers with reviews using paper ID
- Handles missing reviews gracefully
- Extracts title, abstract, and full text from complex JSON structure
- Recursive text extraction as fallback

#### ✅ Review Aggregation
- Supports multiple reviews per paper
- Computes mean score per dimension
- Rounds to nearest integer (1-5)
- Validates all scores are in [1, 5] range

#### ✅ Data Validation
- Removes papers with missing scores
- Removes papers with empty/too-short text (< 100 chars)
- Validates all labels are integers in [1, 5]
- Logs skipped papers with reasons

#### ✅ Sanity Checks
- **Check 1**: Sample counts per split
- **Check 2**: No labels outside [1, 5]
- **Check 3**: No empty texts
- **Check 4**: Example sample display
- **Check 5**: Class distribution analysis with imbalance warnings

#### ✅ Dataset Features
- On-the-fly tokenization (not pre-tokenized)
- Combines title + abstract + full_text
- Converts labels: 1-5 → 0-4 for classification
- Returns proper PyTorch tensors

---

## 🚀 USAGE GUIDE

### Step 1: Install Dependencies

```powershell
pip install torch transformers scikit-learn numpy scipy tqdm tensorboard
```

### Step 2: Verify Data Loading

```powershell
python quick_test.py
```

This will:
- Load all splits (train/dev/test)
- Run sanity checks
- Display class distributions
- Show example samples

### Step 3: Train the Model

```powershell
python train_iclr.py
```

**With custom arguments:**
```powershell
python train_iclr.py `
    --data_path "C:/Facultate/Licenta/data" `
    --output_dir "./outputs_iclr" `
    --base_model "allenai/longformer-base-4096" `
    --batch_size 2 `
    --learning_rate 2e-5 `
    --num_epochs 20
```

### Expected Output

```
================================================================================================ MULTI-TASK ORDINAL CLASSIFICATION FOR PAPER REVIEW SCORING
================================================================================

Device: cuda
Random seed: 42

================================================================================
PHASE 2: LOADING DATA
================================================================================

================================================================================
Loading TRAIN split from C:\Facultate\Licenta\data\train
================================================================================
Found X parsed PDF files

✓ Successfully loaded Y papers

⚠ Skipped papers:
  - no_review: 0
  - invalid_scores: 0
  - empty_text: 0
  - parse_error: 0

... (dev and test splits load similarly)

================================================================================
PHASE 3: RUNNING SANITY CHECKS
================================================================================

✓ Check 1: Sample counts
  Train: X samples
  Dev:   Y samples
  Test:  Z samples
  Total: N samples

✓ Check 2: Label validation
  ✓ All labels in valid range [1, 5]

✓ Check 3: Text content validation
  ✓ All papers have valid text content

✓ Check 4: Example sample
  Paper ID: 104
  Title: Bridge Text and Knowledge by Learning...
  Abstract length: 543 chars
  Full text length: 15234 chars
  Scores:
    IMPACT: 3
    SUBSTANCE: 4
    ...

✓ Check 5: Class distribution
IMPACT:
  Score 1:   10 ( 5.0%) ██
  Score 2:   30 (15.0%) ███████
  Score 3:   80 (40.0%) ████████████████████
  Score 4:   60 (30.0%) ███████████████
  Score 5:   20 (10.0%) █████

... (all dimensions)

================================================================================
✅ ALL SANITY CHECKS PASSED!
================================================================================

... (training begins)
```

---

## 📊 Data Format

### Input JSON Structure

**Parsed PDF** (`data/train/parsed_pdfs/104.pdf.json`):
```json
{
  "name": "104.pdf",
  "metadata": {
    "title": "Paper Title",
    "sections": [
      {
        "heading": "Introduction",
        "text": "Full section text..."
      }
    ]
  }
}
```

**Review** (`data/train/reviews/104.json`):
```json
{
  "reviews": [
    {
      "IMPACT": "3",
      "SUBSTANCE": "4",
      "APPROPRIATENESS": "5",
      "MEANINGFUL_COMPARISON": "2",
      "SOUNDNESS_CORRECTNESS": "4",
      "ORIGINALITY": "3",
      "CLARITY": "3",
      "RECOMMENDATION": "3"
    }
  ]
}
```

### Output Format

**Model predictions** (per paper):
```python
{
    "IMPACT": 4,              # Integers 1-5
    "SUBSTANCE": 5,
    "APPROPRIATENESS": 4,
    "MEANINGFUL_COMPARISON": 3,
    "SOUNDNESS_CORRECTNESS": 4,
    "ORIGINALITY": 4,
    "CLARITY": 3,
    "RECOMMENDATION": 4
}
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'torch'"
```powershell
pip install torch transformers scikit-learn numpy scipy tqdm tensorboard
```

### Issue: "FileNotFoundError: Data path does not exist"
- Check that `data/` folder exists
- Verify it contains `train/`, `dev/`, `test/` subdirectories
- Each subdirectory should have `parsed_pdfs/` and `reviews/` folders

### Issue: "All papers skipped - no valid data"
- Check JSON file format in `parsed_pdfs/` and `reviews/`
- Verify paper IDs match (e.g., `104.pdf.json` and `104.json`)
- Check that reviews contain all required score dimensions

### Issue: "CUDA out of memory"
```powershell
python train_iclr.py --batch_size 1 --base_model "distilbert-base-uncased"
```

---

## 📈 Expected Performance

With proper training data:
- **QWK**: 0.60-0.80 (good to excellent agreement)
- **Accuracy**: 50-70% (exact match is hard for 5 classes)
- **MAE**: 0.3-0.6 (within 1 point on average)
- **Spearman**: 0.7-0.9 (strong rank correlation)

---

## ✅ Implementation Complete Checklist

### Phase 1: Self-Audit
- [x] Verify ordinal classification (NOT regression)
- [x] Verify multi-task setup with 8 dimensions
- [x] Verify CrossEntropyLoss (NO MSELoss)
- [x] Verify all metrics implemented (Accuracy, F1, QWK, Spearman)
- [x] Verify training logic (early stopping, gradient clipping, AdamW)

### Phase 2: Data Loading
- [x] Create `ICLRDataLoader` class
- [x] Load from folder structure (train/dev/test)
- [x] Match papers with reviews by ID
- [x] Extract title, abstract, full text
- [x] Aggregate multiple reviews (mean + round)
- [x] Validate all scores in [1, 5]
- [x] Remove invalid samples
- [x] Create PyTorch `Dataset` class
- [x] On-the-fly tokenization
- [x] Proper label conversion (1-5 → 0-4)

### Phase 3: Sanity Checks
- [x] Check sample counts
- [x] Validate no labels outside [1, 5]
- [x] Validate no empty texts
- [x] Display example samples
- [x] Analyze class distributions
- [x] Warn on severe imbalance

### Additional
- [x] Updated training script
- [x] Test script
- [x] Comprehensive documentation
- [x] Clear error messages

---

## 🎯 Next Steps

1. **Verify Setup**:
   ```powershell
   python quick_test.py
   ```

2. **Start Training**:
   ```powershell
   python train_iclr.py
   ```

3. **Monitor Progress**:
   ```powershell
   tensorboard --logdir ./logs
   ```

4. **Evaluate Results**:
   - Check `outputs/best_model.pt`
   - Check `outputs/test_results.pt`
   - Review confusion matrices

---

## 📝 Summary

✅ **PHASE 1 COMPLETE**: Implementation verified as correct ordinal classification
✅ **PHASE 2 COMPLETE**: Robust data loading implemented
✅ **PHASE 3 COMPLETE**: Comprehensive sanity checks implemented

**The system is ready for training!**

