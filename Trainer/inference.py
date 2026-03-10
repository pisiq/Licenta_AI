"""
inference.py  -  Grade a new paper using the trained model.

Usage
-----
  # Grade a paper from a plain .txt file:
  python inference.py --paper path/to/paper.txt

  # Grade from a PeerRead parsed-PDF JSON:
  python inference.py --paper path/to/173.pdf.json

  # Use a specific checkpoint instead of the best model:
  python inference.py --paper paper.txt --model_path outputs/checkpoint_epoch_5.pt

  # Pretty-print as JSON:
  python inference.py --paper paper.txt --json

How it works
------------
  1. Load best_model.pt (or --model_path)
  2. Read the paper text (plain .txt or PeerRead JSON)
  3. Build input as  TITLE / ABSTRACT / PAPER  (no review - pure inference)
  4. Tokenise with allenai/longformer-base-4096 (max 4096 tokens)
  5. Print predicted scores for all 8 dimensions (1-5 scale)
"""

import os
import sys
import re
import json
import argparse
import torch
from transformers import AutoTokenizer

from config import ModelConfig, TrainingConfig, DataConfig
from model import MultiTaskOrdinalClassifier
from data_preprocessing import TextPreprocessor, _build_paper_only_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str, model_config: ModelConfig, device: torch.device):
    """Load a saved checkpoint and return the model in eval mode."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = MultiTaskOrdinalClassifier(
        base_model_name=model_config.base_model_name,
        score_dimensions=model_config.score_dimensions,
        num_classes=model_config.num_classes,
        dropout=model_config.hidden_dropout_prob,
        use_regression=model_config.use_regression,
    )

    # Checkpoint stores state dict either at top level or under 'model_state_dict'
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def _read_paper(paper_path: str, preprocessor: TextPreprocessor):
    """
    Read a paper and return (title, abstract, body).

    Supported formats
    -----------------
    .txt      Plain text - treated entirely as body text.
    .json /
    .pdf.json PeerRead parsed-PDF JSON with metadata.title / abstractText / sections.
    """
    lower = paper_path.lower()

    if lower.endswith(".txt"):
        with open(paper_path, encoding="utf-8") as f:
            raw = f.read()
        return "", "", preprocessor.preprocess(raw)

    if lower.endswith(".json"):
        with open(paper_path, encoding="utf-8") as f:
            data = json.load(f)
        meta     = data.get("metadata", data)
        title    = preprocessor.clean_text(meta.get("title", "") or "")
        abstract = preprocessor.clean_text(
            meta.get("abstractText", "") or meta.get("abstract", "") or ""
        )
        sections = meta.get("sections") or data.get("sections") or []
        parts = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            heading = sec.get("heading") or ""
            text    = sec.get("text") or ""
            if re.search(r"\breferences?\b", heading, re.IGNORECASE):
                continue
            parts.append(f"### {heading}\n{text}" if heading else text)
        raw_body = "\n\n".join(parts) or meta.get("text", "") or ""
        return title, abstract, preprocessor.preprocess(raw_body)

    raise ValueError(
        f"Unsupported file: {paper_path}\n"
        "Use a .txt file or a PeerRead .pdf.json file."
    )


def _bar(score: float) -> str:
    """Visual 1-5 bar for a predicted score."""
    r = max(1, min(5, round(score)))
    return f"{score:.2f}  [{'#'*r}{'-'*(5-r)}]  ({r}/5)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grade a scientific paper using the trained PeerRead model."
    )
    parser.add_argument("--paper", required=True,
                        help="Path to paper file (.txt or PeerRead .pdf.json)")
    parser.add_argument("--model_path", default=None,
                        help="Checkpoint .pt file. Defaults to <output_dir>/best_model.pt")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    # --- config -----------------------------------------------------------
    model_config    = ModelConfig()
    training_config = TrainingConfig()
    data_config     = DataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- resolve model path -----------------------------------------------
    model_path = args.model_path or os.path.join(
        training_config.output_dir, "best_model.pt"
    )
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train first:  python train.py --use_all_data")
        sys.exit(1)

    # --- load tokenizer + model -------------------------------------------
    print(f"Loading tokenizer: {model_config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
    model     = _load_model(model_path, model_config, device)

    # --- read paper -------------------------------------------------------
    if not os.path.exists(args.paper):
        print(f"ERROR: Paper not found: {args.paper}")
        sys.exit(1)

    preprocessor = TextPreprocessor(
        normalize_whitespace=True,
        remove_references=True,
        max_length=data_config.max_paper_length,
        min_length=0,
    )

    print(f"\nReading paper: {args.paper}")
    title, abstract, body = _read_paper(args.paper, preprocessor)
    print(f"  Title   : {title[:80] or '(not detected)'}")
    print(f"  Abstract: {len(abstract)} chars")
    print(f"  Body    : {len(body)} chars")

    if len(body) < 50:
        print("WARNING: very short body - predictions may be unreliable.")

    # --- tokenise (paper only, no review) ---------------------------------
    input_text = _build_paper_only_text(title, abstract, body)
    encoding   = tokenizer(
        input_text,
        max_length=model_config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    print(f"  Tokens  : {int(attention_mask.sum())}/{model_config.max_length}")

    # --- predict ----------------------------------------------------------
    print("\nRunning model...")
    with torch.no_grad():
        outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
        raw_preds = outputs["predictions"]   # dict dim -> Tensor[1]

    scores = {
        dim: float(raw_preds[dim].squeeze().cpu().item())
        for dim in model_config.score_dimensions
    }

    # --- output -----------------------------------------------------------
    if args.json:
        result = {
            "paper":          args.paper,
            "title":          title,
            "scores":         {d: round(v, 3) for d, v in scores.items()},
            "scores_rounded": {d: round(v)    for d, v in scores.items()},
        }
        print(json.dumps(result, indent=2))
    else:
        W = 60
        print("\n" + "=" * W)
        print("  PREDICTED REVIEW SCORES")
        print(f"  Paper : {os.path.basename(args.paper)}")
        if title:
            print(f"  Title : {title[:W-10]}")
        print("=" * W)
        print(f"  {'Dimension':<28}  Score")
        print(f"  {'-'*28}  {'------'}")
        for dim, score in scores.items():
            print(f"  {dim:<28}  {_bar(score)}")
        avg = sum(scores.values()) / len(scores)
        print("=" * W)
        print(f"  {'AVERAGE':<28}  {_bar(avg)}")
        print("=" * W)
        print("\n  Scale: 1=poor  3=average  5=excellent\n")


if __name__ == "__main__":
    main()
