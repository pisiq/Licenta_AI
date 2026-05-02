from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT_DIR / "Trainer"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(TRAINER_DIR))

from Trainer.config import DataConfig, ModelConfig
from Trainer.generate_review import generate_review_text, pdf_to_json, predict_scores

OUTPUTS_DIR = ROOT_DIR / "outputs"
SCORING_MODEL_PATH = Path(os.getenv("SCORING_MODEL_PATH", OUTPUTS_DIR / "best_model.pt"))

_GEN_CANDIDATES = [
	OUTPUTS_DIR / "review_gen_fast" / "best_review_gen_model",
	OUTPUTS_DIR / "review_gen" / "best_review_gen_model",
]
GEN_MODEL_PATH = Path(os.getenv("GEN_MODEL_PATH", _GEN_CANDIDATES[0]))
if not GEN_MODEL_PATH.exists():
	for candidate in _GEN_CANDIDATES:
		if candidate.exists():
			GEN_MODEL_PATH = candidate
			break


def _now_ms() -> int:
	return int(time.perf_counter() * 1000)


def _device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_json_temp(data: Dict[str, Any]) -> str:
	fd, path = tempfile.mkstemp(suffix=".json", prefix="paper_")
	os.close(fd)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	return path


def score_from_json_source(parsed_json: Dict[str, Any] | None, json_path: str | None) -> Tuple[Dict[str, Any], Dict[str, int]]:
	"""Run scoring model from parsed json payload or file path."""
	started = _now_ms()
	temp_json_path = None
	try:
		if parsed_json is not None:
			temp_json_path = _write_json_temp(parsed_json)
			effective_json_path = temp_json_path
		elif json_path:
			effective_json_path = json_path
		else:
			raise ValueError("Either parsed_json or json_path must be provided.")

		model_config = ModelConfig()
		data_config = DataConfig()
		scores, title, abstract, body = predict_scores(
			json_path=effective_json_path,
			model_path=str(SCORING_MODEL_PATH),
			model_config=model_config,
			data_config=data_config,
			device=_device(),
		)

		return {
			"title": title,
			"abstract": abstract,
			"body": body,
			"scores": scores,
		}, {"total": _now_ms() - started}
	finally:
		if temp_json_path and os.path.exists(temp_json_path):
			os.remove(temp_json_path)


def score_from_pdf_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> Tuple[Dict[str, Any], Dict[str, int]]:
	"""Parse PDF into JSON, then run scoring model."""
	started = _now_ms()
	tmp_dir = tempfile.mkdtemp(prefix="pdf_api_")
	pdf_path = os.path.join(tmp_dir, filename)
	json_path = os.path.join(tmp_dir, "parsed.json")

	try:
		with open(pdf_path, "wb") as f:
			f.write(pdf_bytes)

		parse_start = _now_ms()
		pdf_to_json(pdf_path, json_path)
		parse_time = _now_ms() - parse_start

		infer_start = _now_ms()
		scored, _ = score_from_json_source(parsed_json=None, json_path=json_path)
		infer_time = _now_ms() - infer_start

		return scored, {
			"parse": parse_time,
			"score": infer_time,
			"total": _now_ms() - started,
		}
	finally:
		for p in (pdf_path, json_path):
			if os.path.exists(p):
				os.remove(p)
		if os.path.isdir(tmp_dir):
			os.rmdir(tmp_dir)


def generate_review(
	title: str,
	abstract: str,
	body: str,
	scores: Dict[str, float],
	max_new_tokens: int,
	num_beams: int,
) -> Tuple[str, Dict[str, int]]:
	"""Run review generation model using paper text and score dimensions."""
	started = _now_ms()
	review_text = generate_review_text(
		title=title,
		abstract=abstract,
		body=body,
		scores=scores,
		gen_model_path=str(GEN_MODEL_PATH),
		device=_device(),
		max_new_tokens=max_new_tokens,
		num_beams=num_beams,
	)
	return review_text, {"total": _now_ms() - started}


def model_version() -> str:
	scoring_name = SCORING_MODEL_PATH.name
	generator_name = GEN_MODEL_PATH.name if GEN_MODEL_PATH else "unknown"
	return f"scoring={scoring_name};generator={generator_name}"



