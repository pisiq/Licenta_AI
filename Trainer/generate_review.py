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
from Trainer.pdf_parser import parse_pdf_to_json_file
# NOTE: review_parser is no longer used. The generator now produces a single
# summary paragraph; the output is rendered as-is. Kept on disk for possible
# future structured experiments.

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_SCORING_MODEL = os.path.join("outputs", "best_model.pt")
DEFAULT_GEN_MODEL     = os.path.join("outputs", "review_gen", "best_review_gen_model")
DEFAULT_GEN_MODEL_HF  = "razent/SciFive-base-Pubmed_PMC"   # fallback if no fine-tuned model

PAPER_BODY_CHARS  = 3_500
MAX_INPUT_TOKENS  = 1024
MAX_NEW_TOKENS    = 384   # was 512 — shorter cap fights runaway repetition

# Headers that we strip from the output. The model is trained to stop after
# the summary section, but in case it occasionally emits a leading
# "SUMMARY:" prefix or starts a trailing section, we trim defensively.
_LEADING_SUMMARY_RE = __import__("re").compile(r"^\s*SUMMARY\s*:\s*", __import__("re").IGNORECASE)
_TRAILING_HEADERS    = ("STRENGTHS:", "WEAKNESSES:", "QUESTIONS:",
                        "Strengths:", "Weaknesses:", "Questions:",
                        "STRENGTHS\n", "WEAKNESSES\n", "QUESTIONS\n")


def _trim_to_summary(text: str) -> str:
    """Drop leading 'SUMMARY:' prefix and anything after the first trailing
    section header (STRENGTHS/WEAKNESSES/QUESTIONS). Belt-and-suspenders for
    cases where the model leaks part of the prompt template."""
    text = _LEADING_SUMMARY_RE.sub("", text)
    cut = len(text)
    for h in _TRAILING_HEADERS:
        i = text.find(h)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].rstrip(" \n\t-••:") or text


# ===========================================================================
# Step 1: Parse PDF → JSON (using pymupdf4llm)
# ===========================================================================

def pdf_to_json(pdf_path: str, out_json_path: str) -> str:
    """Parse a PDF using the centralized pdf_parser (font-size title detection,
    author/affiliation filtering, markdown-bold stripping). Writes the
    Science-Parse-compatible JSON to `out_json_path` and returns that path.

    Note: pdf_parser writes to `metadata.abstractText`. _read_paper in
    inference.py already reads either `abstractText` or `abstract`, so the
    rest of the pipeline doesn't need to change.
    """
    print(f"  Parsing PDF: {pdf_path}")
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    parse_pdf_to_json_file(pdf_path, out_json_path)
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
        reg_preds = outputs.get("regression_predictions")

    scores = {}
    for dim, pred in raw_preds.items():
        reg_pred = reg_preds[dim] if reg_preds is not None else None
        pred_class = model.resolve_class_predictions(pred, reg_pred).item()
        scores[dim] = float(pred_class)
    return scores, title, abstract, body


# ===========================================================================
# Step 3: Generate review text with FLAN-T5
# ===========================================================================

def _build_gen_prompt(
    title: str,
    abstract: str,
    body: str,
    scores: Dict[str, float],
    num_classes: int = 5,
    venue: str = "an academic venue",
) -> str:
    """Structured-review prompt — mirrors the training-time prompt.

    The training targets are only the summary section, so the model will
    produce the summary content and stop. We keep the structured prompt
    because empirically it produces more focused output than a bare
    "write a summary" prompt.
    """
    K = int(num_classes)
    scores_parts = [
        f"{dim}: {val:.2f}/{K}"
        for dim, val in scores.items()
        if val is not None
    ]
    scores_str = "  |  ".join(scores_parts)

    return (
        f"Write a structured peer review for the following paper from {venue}. "
        f"Use this exact format (do not add other sections):\n"
        f"SUMMARY: <one paragraph summary>\n"
        f"STRENGTHS:\n- <strength bullet>\n- <strength bullet>\n"
        f"WEAKNESSES:\n- <weakness bullet>\n- <weakness bullet>\n"
        f"QUESTIONS:\n- <question for authors>\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract[:1500]}\n\n"
        f"Paper (excerpt): {body[:PAPER_BODY_CHARS]}\n\n"
        f"Review scores: {scores_str}"
    )


def generate_review_text(
    title:     str,
    abstract:  str,
    body:      str,
    scores:    Dict[str, float],
    gen_model_path: str,
    device:    torch.device,
    *,
    max_new_tokens:        int   = MAX_NEW_TOKENS,
    min_new_tokens:        int   = 80,           # was 120 — let it stop earlier when done
    num_beams:             int   = 1,
    top_p:                 float = 0.90,         # was 0.92 — slightly tighter nucleus
    top_k:                 int   = 50,
    temperature:           float = 0.80,         # was 0.85 — slightly cooler
    repetition_penalty:    float = 1.30,         # was 1.18 — penalize repeats harder
    no_repeat_ngram_size:  int   = 5,            # was 4 — block longer repeated phrases
    seed:                  Optional[int] = None,
    num_classes:           int   = 5,
) -> str:
    """
    Load the fine-tuned FLAN-T5 model and generate a review text.

    Default decoding is nucleus sampling with a repetition penalty — beam
    search is the largest single source of repetitive output for fine-tuned
    seq2seq models on long-form text. Pass `--num_beams > 1` to fall back to
    beam search for A/B comparison.

    Falls back to google/flan-t5-base if the fine-tuned model is not found.
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

    prompt = _build_gen_prompt(title, abstract, body, scores, num_classes=num_classes)
    enc    = tokenizer(
        prompt,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    use_sampling = num_beams <= 1
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )
    if use_sampling:
        gen_kwargs.update(
            do_sample=True,
            num_beams=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        print(f"  Decoding: nucleus sampling  (top_p={top_p}, top_k={top_k}, T={temperature}, "
              f"rep_pen={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size}, "
              f"min/max new tokens={min_new_tokens}/{max_new_tokens})")
    else:
        gen_kwargs.update(
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
        )
        print(f"  Decoding: beam search  (beams={num_beams}, "
              f"rep_pen={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size})")

    with torch.no_grad():
        generated_ids = gen_model.generate(**gen_kwargs)

    review_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return review_text


# ===========================================================================
# Print helpers
# ===========================================================================

def print_scores(scores: Dict[str, float], title: str, json_path: str, num_classes: int = 5):
    W = 62
    K = int(num_classes)
    print("\n" + "=" * W)
    print("  PREDICTED REVIEW SCORES")
    print(f"  Paper : {os.path.basename(json_path)}")
    if title:
        print(f"  Title : {title[:W-10]}")
    print("=" * W)
    print(f"  {'Dimension':<28}  Score")
    print(f"  {'-'*28}  {'------'}")
    for dim, score in scores.items():
        print(f"  {dim:<28}  {_bar(score, K)}")
    avg = sum(scores.values()) / len(scores)
    print("=" * W)
    print(f"  {'AVERAGE':<28}  {_bar(avg, K)}")
    print("=" * W)
    rec = scores.get("RECOMMENDATION", 0.0)
    print(f"\n  Scale: 1 (poor)  ...  {K} (excellent)")
    print(f"  Primary score (RECOMMENDATION): {rec:.3f} / {K}\n")


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
    p.add_argument("--max_new_tokens",       type=int,   default=MAX_NEW_TOKENS)
    p.add_argument("--min_new_tokens",       type=int,   default=80,
                   help="Force the model to produce at least this many new tokens.")
    p.add_argument("--num_beams",            type=int,   default=1,
                   help="1 = sampling (default). >1 falls back to beam search.")
    p.add_argument("--top_p",                type=float, default=0.90,
                   help="Nucleus sampling top-p. Only used when --num_beams==1.")
    p.add_argument("--top_k",                type=int,   default=50,
                   help="Top-k sampling. Only used when --num_beams==1.")
    p.add_argument("--temperature",          type=float, default=0.80,
                   help="Sampling temperature. Only used when --num_beams==1.")
    p.add_argument("--repetition_penalty",   type=float, default=1.30,
                   help=">1.0 discourages token repetition. 1.0 = off. 1.3 is fairly strict.")
    p.add_argument("--no_repeat_ngram_size", type=int,   default=5,
                   help="Disallow repeating n-grams of this size in the output.")
    p.add_argument("--seed",                 type=int,   default=None,
                   help="Random seed for sampling reproducibility.")
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
    print_scores(scores, title, json_path, num_classes=model_config.num_classes)

    if args.no_generate:
        return

    # ---- Step 3: Generate review text -------------------------------------
    print("[Step 2/2] Generating structured review with SciFive/FLAN-T5...")
    review_text = generate_review_text(
        title=title,
        abstract=abstract,
        body=body,
        scores=scores,
        gen_model_path=args.gen_model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
        num_classes=model_config.num_classes,
    )

    # Summary-only rendering: the entire model output IS the summary.
    # Safety trim removes any stray prompt-template leakage.
    review_text = _trim_to_summary(review_text)
    W = 70
    print("\n" + "=" * W)
    print("  GENERATED SUMMARY")
    print("=" * W)
    print(review_text.strip())
    print("=" * W)

    # ---- Save output -------------------------------------------------------
    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join("outputs", f"{base}_review.txt")

    K = model_config.num_classes
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== PREDICTED SCORES ===\n")
        for dim, score in scores.items():
            f.write(f"  {dim}: {score:.3f}/{K}\n")
        f.write("\n=== GENERATED SUMMARY ===\n\n")
        f.write(review_text.strip())
        f.write("\n")
    print(f"\n  Summary saved → {out_path}")


if __name__ == "__main__":
    main()

