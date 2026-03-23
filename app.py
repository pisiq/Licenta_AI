"""
app.py  -  Interfata Gradio pentru evaluarea lucrarilor stiintifice.

Pipeline: PDF → parsare → scoruri (best_model.pt) → review text (model.safetensors)

Rulare:
    python app.py
"""
from __future__ import annotations

import os
import sys
import json
import tempfile

# ---------------------------------------------------------------------------
# Asigura importabilitatea pachetului Trainer
# ---------------------------------------------------------------------------
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAINER_DIR = os.path.join(ROOT_DIR, "Trainer")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TRAINER_DIR)

import gradio as gr
import torch

from Trainer.config import ModelConfig, DataConfig
from Trainer.generate_review import (
    pdf_to_json,
    predict_scores,
    generate_review_text,
)

# ---------------------------------------------------------------------------
# Cai implicite catre modele
# ---------------------------------------------------------------------------
DEFAULT_SCORING_MODEL = os.path.join(ROOT_DIR, "outputs", "best_model.pt")
DEFAULT_GEN_MODEL     = os.path.join(ROOT_DIR, "outputs", "review_gen_fast", "best_review_gen_model")

SCORE_LABELS = {
    "RECOMMENDATION":        "Recomandare generala",
    "IMPACT":                "Impact",
    "SUBSTANCE":             "Substanta",
    "APPROPRIATENESS":       "Relevanta",
    "MEANINGFUL_COMPARISON": "Comparatie relevanta",
    "SOUNDNESS_CORRECTNESS": "Corectitudine stiintifica",
    "ORIGINALITY":           "Originalitate",
    "CLARITY":               "Claritate",
}

SCORE_COLORS = {
    1: "#ef4444",
    2: "#f97316",
    3: "#eab308",
    4: "#22c55e",
    5: "#16a34a",
}


# ---------------------------------------------------------------------------
# Utilitare de formatare
# ---------------------------------------------------------------------------

def _score_bar(score: float) -> str:
    """Bara vizuala pentru un scor intre 1 si 5."""
    r = max(1, min(5, round(score)))
    filled   = "█" * r
    empty    = "░" * (5 - r)
    return f"{filled}{empty}  {score:.2f}/5.0"


def _format_scores_markdown(scores: dict[str, float], title: str) -> str:
    lines = []
    if title:
        lines.append(f"### {title}\n")

    lines.append("| Dimensiune | Scor | Vizualizare |")
    lines.append("|:-----------|:----:|:------------|")

    for dim, score in scores.items():
        label = SCORE_LABELS.get(dim, dim)
        r     = max(1, min(5, round(score)))
        bar   = "★" * r + "☆" * (5 - r)
        lines.append(f"| **{label}** | {score:.2f} | {bar} |")

    avg = sum(scores.values()) / len(scores)
    r   = max(1, min(5, round(avg)))
    bar = "★" * r + "☆" * (5 - r)
    lines.append(f"| **MEDIE** | **{avg:.2f}** | {bar} |")

    lines.append(f"\n> Scara: 1 = slab  |  3 = mediu  |  5 = excelent")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functii de inferenta
# ---------------------------------------------------------------------------

def _run_scoring(pdf_file) -> tuple[dict, str, str, str, str]:
    """
    Parsează PDF-ul si ruleaza modelul de scoruri.
    Returneaza (scores, title, abstract, body, json_path).
    """
    if pdf_file is None:
        raise gr.Error("Te rog incarca un fisier PDF.")

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()
    data_config  = DataConfig()

    # Salveaza JSON-ul parsat intr-un director temporar
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    out_dir   = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{base_name}_parsed.json")

    pdf_to_json(pdf_file, json_path)

    scores, title, abstract, body = predict_scores(
        json_path=json_path,
        model_path=DEFAULT_SCORING_MODEL,
        model_config=model_config,
        data_config=data_config,
        device=device,
    )
    return scores, title, abstract, body, json_path


def run_pipeline(pdf_file, use_model1, use_model2, num_beams, max_new_tokens):
    """
    Callback principal. Ruleaza modelele bifate.
    Returneaza (scores_md, review_text, scores_visible, review_visible).
    """
    if not use_model1 and not use_model2:
        raise gr.Error("Bifeaza cel putin un model.")

    NUM_BEAMS = 8
    MAX_NEW_TOKENS = 1024
    try:
        scores, title, abstract, body, _ = _run_scoring(pdf_file)
        scores_md = _format_scores_markdown(scores, title or "(titlu nedectat)") if use_model1 else ""

        review_text = ""
        if use_model2:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            review_text = generate_review_text(
                title=title,
                abstract=abstract,
                body=body,
                scores=scores,
                gen_model_path=DEFAULT_GEN_MODEL,
                device=device,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )

        return (
            scores_md,
            review_text,
            gr.update(visible=use_model1),
            gr.update(visible=use_model2),
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Eroare la procesare: {e}") from e


def toggle_review_params(use_model2):
    """Arata/ascunde parametrii de generare in functie de checkbox."""
    return gr.update(visible=use_model2)


# ---------------------------------------------------------------------------
# Interfata Gradio
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Evaluator Lucrare Stiintifica",
        theme=gr.themes.Soft(),
        css=".header { text-align: center; margin-bottom: 1rem; }",
    ) as demo:

        # --- Header ---
        gr.Markdown(
            """
            # Evaluator Automat de Lucrari Stiintifice
            Incarca un PDF si alege ce modele vrei sa rulezi.
            """,
            elem_classes="header",
        )

        with gr.Row():
            # ----------------------------------------------------------------
            # Coloana stanga – input + optiuni
            # ----------------------------------------------------------------
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="Incarca PDF",
                    file_types=[".pdf"],
                )

                gr.Markdown("### Modele")
                use_model1 = gr.Checkbox(
                    label="Genereaza scoruri pentru lucrare",
                    value=False,
                )
                use_model2 = gr.Checkbox(
                    label="Genereaza un revenzie pentru lucrare",
                    value=False,
                )

                btn_run = gr.Button("Ruleaza", variant="primary")

            # ----------------------------------------------------------------
            # Coloana dreapta – rezultate
            # ----------------------------------------------------------------
            with gr.Column(scale=2):
                score_output = gr.Markdown(
                    label="Scoruri (Model 1)",
                    value="*Rezultatele vor aparea aici dupa analiza.*",
                    visible=True,
                )
                review_output = gr.Textbox(
                    label="Review generat (Model 2)",
                    placeholder="Textul review-ului academic generat va aparea aici...",
                    lines=20,
                    visible=False,
                )






        # --- Buton principal ---
        btn_run.click(
            fn=run_pipeline,
            inputs=[pdf_input, use_model1, use_model2 ],
            outputs=[score_output, review_output, score_output, review_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Scoring model : {DEFAULT_SCORING_MODEL}")
    print(f"Generator model: {DEFAULT_GEN_MODEL}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
