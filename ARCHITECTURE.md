# 🏗️ Architecture Visualization

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCIENTIFIC PAPER REVIEW SCORER                        │
│                     Multi-Task Ordinal Classification                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT: Scientific Paper                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Title: "Deep Learning for NLP"                                           │
│ • Abstract: "We propose a novel approach..."                               │
│ • Full Text: "Introduction... Methods... Results..."                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING PIPELINE                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Clean PDF artifacts (remove \x00, control chars)                        │
│ 2. Normalize whitespace (collapse multiple spaces)                         │
│ 3. Remove references section (optional)                                    │
│ 4. Truncate to max length (10,000 chars)                                   │
│ 5. Combine: title + abstract + full_text                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOKENIZATION                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Tokenizer: Longformer/BERT/SciBERT                                       │
│ • Max Length: 4096 tokens (configurable)                                   │
│ • Padding: max_length                                                       │
│ • Truncation: enabled                                                       │
│ • Output: input_ids [1, 4096], attention_mask [1, 4096]                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRANSFORMER ENCODER (Shared)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────┐   │
│ │  Longformer-base-4096 (or other transformer)                        │   │
│ │  • 12 layers, 768 hidden size, 12 attention heads                   │   │
│ │  • Efficient attention for long sequences                           │   │
│ │  • Pre-trained on scientific papers                                 │   │
│ └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│ Input:  [batch, 4096] tokens                                               │
│ Output: [batch, 4096, 768] hidden states                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ POOLING                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Strategy: CLS token or Mean pooling                                      │
│ • Output: [batch, 768] pooled representation                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION HEADS (8 separate heads, one per dimension)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────┐   │
│  │   IMPACT      │  │  SUBSTANCE    │  │ APPROPRIATENESS│  │   ...    │   │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤  ├──────────┤   │
│  │ Dropout(0.1)  │  │ Dropout(0.1)  │  │ Dropout(0.1)  │  │ Dropout  │   │
│  │ Linear(768→5) │  │ Linear(768→5) │  │ Linear(768→5) │  │ Linear   │   │
│  │ Softmax       │  │ Softmax       │  │ Softmax       │  │ Softmax  │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────┘   │
│        ↓                  ↓                  ↓                  ↓          │
│  [batch, 5]         [batch, 5]         [batch, 5]         [batch, 5]      │
│  Probs for          Probs for          Probs for          Probs for       │
│  scores 1-5         scores 1-5         scores 1-5         scores 1-5      │
│                                                                             │
│  8 Dimensions Total:                                                        │
│  • IMPACT                      • SOUNDNESS_CORRECTNESS                      │
│  • SUBSTANCE                   • ORIGINALITY                                │
│  • APPROPRIATENESS             • CLARITY                                    │
│  • MEANINGFUL_COMPARISON       • RECOMMENDATION                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LOSS COMPUTATION (Training)                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each dimension:                                                         │
│   • Cross-Entropy Loss (predicted_probs, true_label)                       │
│   • Optional: Weighted by class frequencies                                │
│                                                                             │
│ Total Loss = Average of all 8 dimension losses                             │
│ Backpropagate through shared encoder + all heads                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Predicted Scores                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ {                                                                           │
│   "IMPACT": 4,                       (argmax of [0.05, 0.10, 0.15, 0.60,  │
│   "SUBSTANCE": 5,                                 0.10])                    │
│   "APPROPRIATENESS": 3,                                                     │
│   "MEANINGFUL_COMPARISON": 4,                                               │
│   "SOUNDNESS_CORRECTNESS": 5,                                               │
│   "ORIGINALITY": 4,                                                         │
│   "CLARITY": 3,                                                             │
│   "RECOMMENDATION": 4                                                       │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Raw Data   │ ───> │ Preprocessing│ ───> │  Dataset    │
│  (JSON)     │      │              │      │  (PyTorch)  │
└─────────────┘      └──────────────┘      └─────────────┘
                                                  │
                                                  ↓
                           ┌────────────────────────────────┐
                           │     Train/Dev/Test Split       │
                           │     (80% / 10% / 10%)          │
                           └────────────────────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ↓                  ↓                  ↓
              ┌────────────┐    ┌────────────┐    ┌────────────┐
              │   Train    │    │    Dev     │    │   Test     │
              │  DataLoader│    │ DataLoader │    │ DataLoader │
              └────────────┘    └────────────┘    └────────────┘
                     │                  │                  │
                     ↓                  │                  │
         ┌───────────────────┐         │                  │
         │  Training Loop    │         │                  │
         │                   │         │                  │
         │  For each epoch:  │         │                  │
         │  1. Forward pass  │         │                  │
         │  2. Compute loss  │         │                  │
         │  3. Backward pass │         │                  │
         │  4. Update weights│         │                  │
         │  5. Clip gradients│         │                  │
         └───────────────────┘         │                  │
                     │                  │                  │
                     └─────────> ┌─────┴──────┐           │
                                 │  Evaluate  │           │
                                 │  on Dev    │           │
                                 │  Compute   │           │
                                 │  QWK       │           │
                                 └────────────┘           │
                                       │                  │
                                       ↓                  │
                            ┌───────────────────┐         │
                            │ Early Stopping    │         │
                            │ Check             │         │
                            │ (patience = 3)    │         │
                            └───────────────────┘         │
                                       │                  │
                                       ↓                  │
                            ┌───────────────────┐         │
                            │ Save Best Model   │         │
                            │ (highest QWK)     │         │
                            └───────────────────┘         │
                                                           │
                            After Training ────────────────┘
                                       │
                                       ↓
                            ┌───────────────────┐
                            │ Evaluate on Test  │
                            │ - QWK per dim     │
                            │ - Confusion matrix│
                            │ - Accuracy        │
                            │ - F1 score        │
                            └───────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          EVALUATION METRICS FLOW
═══════════════════════════════════════════════════════════════════════════════

Predictions:   [4, 5, 3, 4, 5, 4, 3, 4]  (8 dimensions)
True Labels:   [4, 4, 3, 5, 5, 4, 4, 4]

                       ↓

For EACH dimension (e.g., IMPACT):
    Predictions: [4, 5, 3, 4, 5, ...]
    Labels:      [4, 4, 3, 5, 5, ...]

    ┌──────────────────────────────────────┐
    │  Compute Metrics:                    │
    │  • Accuracy = 70%                    │
    │  • Macro F1 = 0.68                   │
    │  • QWK = 0.75  ⭐ (primary)          │
    │  • Spearman = 0.82                   │
    │  • MAE = 0.45                        │
    │  • Confusion Matrix [5x5]            │
    └──────────────────────────────────────┘

                       ↓

Aggregate across ALL dimensions:
    ┌──────────────────────────────────────┐
    │  Macro Averages:                     │
    │  • Avg Accuracy = 68%                │
    │  • Avg F1 = 0.66                     │
    │  • Avg QWK = 0.73  ⭐                │
    │  • Avg Spearman = 0.79               │
    │  • Avg MAE = 0.48                    │
    └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                           FILE ORGANIZATION
═══════════════════════════════════════════════════════════════════════════════

Licenta/
├── 📄 config.py                     ← Configuration classes
├── 📄 data_preprocessing.py         ← Data loading & cleaning
├── 📄 model.py                      ← Model architecture
├── 📄 metrics.py                    ← Evaluation metrics
├── 📄 trainer.py                    ← Training loop
├── 📄 train.py                      ← Main training script
├── 📄 inference.py                  ← Prediction interface
├── 📄 example.py                    ← End-to-end demo
├── 📄 test_pipeline.py              ← Unit tests
├── 📄 generate_sample_data.py       ← Sample data generator
├── 📄 verify_setup.py               ← Setup verification
├── 📄 requirements.txt              ← Dependencies
├── 📄 README.md                     ← Full documentation
├── 📄 QUICKSTART.md                 ← Quick start guide
├── 📄 PROJECT_SUMMARY.md            ← Technical summary
├── 📄 IMPLEMENTATION_COMPLETE.md    ← Implementation details
├── 📄 .gitignore                    ← Git ignore patterns
└── 📁 outputs/                      ← Saved models (created during training)
    └── 📁 logs/                     ← TensorBoard logs


═══════════════════════════════════════════════════════════════════════════════
                        KEY COMPONENTS INTERACTION
═══════════════════════════════════════════════════════════════════════════════

    ┌──────────────┐
    │   config.py  │ ──────┐
    └──────────────┘       │
                           ↓
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ train.py     │───→│  trainer.py  │───→│  model.py    │
    └──────────────┘    └──────────────┘    └──────────────┘
           │                    │                    │
           ↓                    ↓                    ↓
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ data_        │    │  metrics.py  │    │ inference.py │
    │ preprocessing│    └──────────────┘    └──────────────┘
    └──────────────┘

All components are:
• Modular (can be used independently)
• Well-documented (comprehensive docstrings)
• Tested (unit tests in test_pipeline.py)
• Type-hinted (for better IDE support)
• Configurable (via config.py)


═══════════════════════════════════════════════════════════════════════════════
                              USAGE FLOW
═══════════════════════════════════════════════════════════════════════════════

1. SETUP
   pip install -r requirements.txt
   python verify_setup.py

2. PREPARE DATA
   python generate_sample_data.py
   (or prepare your own JSON)

3. TRAIN
   python train.py --data_path ./data/papers_reviews.json

4. EVALUATE
   (automatic during training)
   Best model saved based on dev QWK

5. PREDICT
   from inference import load_model_for_inference
   predictor = load_model_for_inference("./outputs/best_model.pt")
   scores = predictor.predict_scores(paper_text="...")


═══════════════════════════════════════════════════════════════════════════════
                         🎉 COMPLETE & READY TO USE! 🎉
═══════════════════════════════════════════════════════════════════════════════

