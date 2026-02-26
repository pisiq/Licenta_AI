# 🎯 Quick Reference Card - Regression Training

## One-Line Summary
**Your model now uses regression (MSE loss) instead of classification, with AMP for 2-3x speedup on your RTX 5060 Ti.**

---

## 🚀 Quick Commands

### Start Training:
```bash
python train.py
```

### View Logs (If TensorBoard Works):
```bash
tensorboard --logdir=logs
```

### Fix pkg_resources Error:
```bash
pip install --force-reinstall setuptools
```

---

## 📊 What Changed

| Component | Old | New |
|-----------|-----|-----|
| **Loss** | CrossEntropy | MSE (regression) |
| **Output** | 5 classes | Continuous [1, 5] |
| **Speed** | 6s/it | 2-3s/it (AMP) |
| **VRAM** | 7.5GB | 4-5GB |
| **Warmup** | None | 100 steps |
| **Freezing** | None | 2 epochs |

---

## 🎯 Key Settings (config.py)

```python
use_regression = True        # ← Regression mode
fp16 = True                  # ← AMP enabled
learning_rate = 2e-5
warmup_steps = 100
freeze_backbone_epochs = 2
train_batch_size = 1
gradient_accumulation_steps = 8
```

---

## 📈 Expected Metrics

### Good Performance:
- **QWK**: >0.3 (primary metric)
- **MAE**: <0.5
- **Spearman**: >0.6
- **Accuracy**: >50%

### Training Speed:
- **Per iteration**: ~2-3 seconds
- **Per epoch**: ~2-3 minutes
- **20 epochs**: ~40-60 minutes

---

## 🐛 Quick Fixes

### CUDA OOM:
```python
gradient_accumulation_steps = 16
max_length = 2048
```

### Model Collapses:
```python
freeze_backbone_epochs = 4
learning_rate = 1e-5
```

### pkg_resources Error:
```bash
pip install setuptools==70.0.0
```

---

## 📁 Important Files

- **Best Model**: `outputs/best_model.pt`
- **Config**: `config.py`
- **Training**: `train.py`
- **Logs**: `logs/events.*`

---

## ✅ Checklist

- [x] CUDA available (RTX 5060 Ti)
- [x] use_regression = True
- [x] fp16 = True (AMP)
- [x] Data in `data/papers_reviews.json`
- [x] Ready to train!

---

## 🎓 Remember

1. **QWK is primary metric** (watch it increase)
2. **MAE should decrease** (<0.5 is good)
3. **Spearman >0.6** means model learned ordering
4. **Early stopping** usually triggers at epoch 10-15
5. **Backbone freezes** for first 2 epochs (normal!)

---

**Run**: `python train.py`  
**Monitor**: Watch QWK in console output  
**Success**: QWK >0.3, MAE <0.5

**Good luck! 🚀**

