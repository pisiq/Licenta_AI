"""
generate_review.py
==================
Inferență completă: PDF → scoruri (best_model.pt) + review text (FLAN-T5 fine-tuned).

Folosire
--------
  # Din directorul rădăcină al proiectului:
  python Trainer/generate_review.py --pdf 1.pdf

  # sau cu un JSON deja parsat:
  python Trainer/generate_review.py --json outputs/1.json

  # cu un model generativ specific:
  python Trainer/generate_review.py --pdf 1.pdf \
      --gen_model outputs/review_gen/best_review_gen_model

Cerinte
-------
  pip install pymupdf4llm transformers torch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Make sure Trainer package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Trainer.config import ModelConfig, DataConfig
from Trainer.data_preprocessing import TextPreprocessor, _build_paper_only_text, SCORE_DIMENSIONS
from Trainer.inference import _load_model, _read_paper, _bar

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_SCORING_MODEL = os.path.join("outputs", "best_model.pt")
DEFAULT_GEN_MODEL     = os.path.join("outputs", "review_gen", "best_review_gen_model")
DEFAULT_GEN_MODEL_HF  = "google/flan-t5-base"   # fallback dacă nu ai fine-tuned

PAPER_BODY_CHARS  = 3_500
MAX_INPUT_TOKENS  = 1024
MAX_NEW_TOKENS    = 512


# ===========================================================================
# Step 1: Parse PDF → JSON (using pymupdf4llm)
# ===========================================================================

def pdf_to_json(pdf_path: str, out_json_path: str) -> str:
    """
    Parse a PDF file to the internal JSON format used by the project.
    Returns the path to the saved JSON file.
    """
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError(
            "pymupdf4llm not installed. Run: pip install pymupdf4llm"
        )

    print(f"  Parsing PDF: {pdf_path}")
    md_text: str = pymupdf4llm.to_markdown(pdf_path)

    # -----------------------------------------------------------------------
    # Convert markdown to the project's parsed PDF JSON structure:
    #   { "metadata": {...}, "sections": [{"heading": ..., "text": ...}] }
    # -----------------------------------------------------------------------
    lines   = md_text.split("\n")
    sections = []
    current_heading = "Abstract"
    current_text: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect markdown headings as section breaks
        if stripped.startswith("#"):
            # Save previous section
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "text":    "\n".join(current_text).strip(),
                })
                current_text = []
            current_heading = stripped.lstrip("#").strip()
        else:
            if stripped:
                current_text.append(stripped)

    # Flush last section
    if current_text:
        sections.append({
            "heading": current_heading,
            "text":    "\n".join(current_text).strip(),
        })

    # Try to extract title (first heading or first bold line)
    title = ""
    for sec in sections:
        if sec["heading"] and sec["heading"].lower() not in ("abstract",):
            title = sec["heading"]
            break

    # Build abstract
    abstract = ""
    for sec in sections:
        if "abstract" in sec["heading"].lower():
            abstract = sec["text"]
            break

    parsed = {
        "metadata": {
            "title":    title,
            "abstract": abstract,
        },
        "sections": sections,
    }

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"  JSON saved → {out_json_path}")
    return out_json_path


# ===========================================================================
# Step 2: Predict scores with best_model.pt
# ===========================================================================

def predict_scores(
    json_path:    str,
    model_path:   str,
    model_config: ModelConfig,
    data_config:  DataConfig,
    device:       torch.device,
) -> tuple[Dict[str, float], str, str, str]:
    """
    Returns (scores_dict, title, abstract, body).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
    model     = _load_model(model_path, model_config, device)

    preprocessor = TextPreprocessor(
        normalize_whitespace=True,
        remove_references=True,
        max_length=data_config.max_paper_length,
        min_length=0,
    )

    title, abstract, body = _read_paper(json_path, preprocessor)

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

    with torch.no_grad():
        outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
        raw_preds = outputs["predictions"]

    scores = {
        dim: float(raw_preds[dim].squeeze().cpu().item())
        for dim in model_config.score_dimensions
    }
    return scores, title, abstract, body


# ===========================================================================
# Step 3: Generate review text with FLAN-T5
# ===========================================================================

def _build_gen_prompt(
    title: str,
    abstract: str,
    body: str,
    scores: Dict[str, float],
) -> str:
    scores_parts = [
        f"{dim}: {val:.2f}/5.0"
        for dim, val in scores.items()
        if val is not None
    ]
    scores_str = "  |  ".join(scores_parts)

    return (
        f"Write an academic peer review for the following machine learning paper.\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract[:1500]}\n\n"
        f"Paper (excerpt): {body[:PAPER_BODY_CHARS]}\n\n"
        f"Review scores: {scores_str}\n\n"
        f"Write a detailed review covering: summary, strengths, weaknesses, "
        f"questions for authors, and final recommendation."
    )


def generate_review_text(
    title:     str,
    abstract:  str,
    body:      str,
    scores:    Dict[str, float],
    gen_model_path: str,
    device:    torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_beams:      int = 4,
    temperature:    float = 0.8,
) -> str:
    """
    Load the fine-tuned FLAN-T5 model and generate a review text.
    Falls back to google/flan-t5-base if fine-tuned model is not found.
    """
    if os.path.isdir(gen_model_path):
        print(f"  Loading fine-tuned generator: {gen_model_path}")
        model_name = gen_model_path
    else:
        print(f"  [WARNING] Fine-tuned model not found at: {gen_model_path}")
        print(f"  Falling back to base model: {DEFAULT_GEN_MODEL_HF}")
        print(f"  For better results, run: python Trainer/review_generator_train.py")
        model_name = DEFAULT_GEN_MODEL_HF

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen_model.to(device)
    gen_model.eval()

    prompt = _build_gen_prompt(title, abstract, body, scores)
    enc    = tokenizer(
        prompt,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = gen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=temperature if num_beams == 1 else 1.0,
            do_sample=num_beams == 1,
        )

    review_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return review_text


# ===========================================================================
# Print helpers
# ===========================================================================

def print_scores(scores: Dict[str, float], title: str, json_path: str):
    W = 62
    print("\n" + "=" * W)
    print("  PREDICTED REVIEW SCORES")
    print(f"  Paper : {os.path.basename(json_path)}")
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
    rec = scores.get("RECOMMENDATION", 0.0)
    print(f"\n  Scale: 1=poor  3=average  5=excellent")
    print(f"  Primary score (RECOMMENDATION): {rec:.3f} / 5.0\n")


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full pipeline: PDF/JSON → scores + generated review text"
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pdf",  help="Path to PDF file to parse and evaluate")
    grp.add_argument("--json", help="Path to pre-parsed JSON file")

    p.add_argument("--scoring_model", default=DEFAULT_SCORING_MODEL,
                   help="Path to best_model.pt (scoring model)")
    p.add_argument("--gen_model",     default=DEFAULT_GEN_MODEL,
                   help="Path to fine-tuned FLAN-T5 directory (or HF model name)")
    p.add_argument("--output",        default=None,
                   help="Optional path to save the generated review as .txt")
    p.add_argument("--no_generate",   action="store_true",
                   help="Skip review text generation (only predict scores)")
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--num_beams",      type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- Resolve JSON path ------------------------------------------------
    if args.pdf:
        base_name = os.path.splitext(os.path.basename(args.pdf))[0]
        json_path = os.path.join("outputs", f"{base_name}_parsed.json")
        pdf_to_json(args.pdf, json_path)
    else:
        json_path = args.json

    # ---- Step 2: Predict scores -------------------------------------------
    print("[Step 1/2] Predicting scores with scoring model...")
    model_config = ModelConfig()
    data_config  = DataConfig()

    scores, title, abstract, body = predict_scores(
        json_path=json_path,
        model_path=args.scoring_model,
        model_config=model_config,
        data_config=data_config,
        device=device,
    )
    print_scores(scores, title, json_path)

    if args.no_generate:
        return

    # ---- Step 3: Generate review text -------------------------------------
    print("[Step 2/2] Generating review text with FLAN-T5...")
    review_text = generate_review_text(
        title=title,
        abstract=abstract,
        body=body,
        scores=scores,
        gen_model_path=args.gen_model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    W = 62
    print("\n" + "=" * W)
    print("  GENERATED REVIEW TEXT")
    print("=" * W)
    print(review_text)
    print("=" * W)

    # ---- Save output -------------------------------------------------------
    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join("outputs", f"{base}_review.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== PREDICTED SCORES ===\n")
        for dim, score in scores.items():
            f.write(f"  {dim}: {score:.3f}/5.0\n")
        f.write(f"\n=== GENERATED REVIEW ===\n\n")
        f.write(review_text)
    print(f"\n  Review saved → {out_path}")


if __name__ == "__main__":
    main()

