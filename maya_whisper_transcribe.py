#!/usr/bin/env python3
"""Transcribe a Maya speech corpus with Whisper and compute WER.

This utility expects a CSV file containing at least two columns:
```
audio_path,transcript
```
`audio_path` can be absolute or relative to the CSV file location. The script
will process the audio files sequentially using OpenAI Whisper, saving the
predicted transcription for each audio clip. Progress is persisted in an output
CSV file so that interrupted runs can resume from where they left off.

Example usage:
```
python maya_whisper_transcribe.py \
    --metadata metadata.csv \
    --output predictions.csv \
    --model medium \
    --device cuda
```
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import whisper
from jiwer import wer


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="CSV file with columns 'audio_path' and 'transcript'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV file where predictions will be stored.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name to use (e.g., tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language code to pass to Whisper (default: es).",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for decoding. Larger values may improve accuracy but slow inference.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for decoding (0 for greedy / deterministic).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log progress every N processed audios.",
    )
    parser.add_argument(
        "--compute_wer",
        action="store_true",
        help="Compute and display WER after processing.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def read_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_cols = {"audio_path", "transcript"} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"Metadata file {metadata_path} is missing columns: {sorted(missing_cols)}"
            )
        for row in reader:
            rows.append(row)
    return rows


def load_existing_predictions(output_path: Path) -> Dict[str, Dict[str, str]]:
    if not output_path.exists():
        return {}

    predictions: Dict[str, Dict[str, str]] = {}
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions[row["audio_path"]] = row
    LOGGER.info("Loaded %d existing predictions from %s", len(predictions), output_path)
    return predictions


def ensure_output_writer(output_path: Path) -> Tuple[csv.DictWriter, object]:
    output_exists = output_path.exists()
    handle = output_path.open("a", newline="", encoding="utf-8")
    fieldnames = ["audio_path", "reference_transcript", "predicted_transcript"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not output_exists:
        writer.writeheader()
    return writer, handle


def resolve_audio_path(audio_path: str, metadata_path: Path) -> Path:
    path = Path(audio_path)
    if path.is_file():
        return path
    candidate = metadata_path.parent / path
    return candidate


def transcribe_audio(
    model: whisper.Whisper,
    audio_path: Path,
    device: str,
    language: str,
    beam_size: int,
    temperature: float,
) -> str:
    options = dict(
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        fp16=device.startswith("cuda"),
    )
    result = model.transcribe(str(audio_path), **options)
    return result.get("text", "").strip()


def process_corpus(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    metadata = read_metadata(args.metadata)
    existing = load_existing_predictions(args.output)
    writer, handle = ensure_output_writer(args.output)

    try:
        model = whisper.load_model(args.model, device=args.device)
    except Exception:
        handle.close()
        raise

    processed_rows: List[Tuple[str, str, str]] = []
    try:
        for index, row in enumerate(metadata, start=1):
            audio_rel_path = row["audio_path"]
            reference = row["transcript"]

            if audio_rel_path in existing:
                if index % args.log_interval == 0:
                    LOGGER.info("Skipping %s (already processed)", audio_rel_path)
                continue

            audio_path = resolve_audio_path(audio_rel_path, args.metadata)
            if not audio_path.is_file():
                LOGGER.error("Audio file not found: %s", audio_path)
                continue

            try:
                prediction = transcribe_audio(
                    model=model,
                    audio_path=audio_path,
                    device=args.device,
                    language=args.language,
                    beam_size=args.beam_size,
                    temperature=args.temperature,
                )
            except Exception as exc:
                LOGGER.exception("Failed to transcribe %s: %s", audio_path, exc)
                continue

            writer.writerow(
                {
                    "audio_path": audio_rel_path,
                    "reference_transcript": reference,
                    "predicted_transcript": prediction,
                }
            )
            handle.flush()
            processed_rows.append((audio_rel_path, reference, prediction))

            if index % args.log_interval == 0:
                LOGGER.info("Processed %d/%d files", index, len(metadata))
    finally:
        handle.close()

    return processed_rows


def compute_wer(results: Iterable[Tuple[str, str, str]], output_path: Path) -> float:
    references: List[str] = []
    predictions: List[str] = []

    for _audio_path, reference, prediction in results:
        references.append(reference)
        predictions.append(prediction)

    if not references:
        LOGGER.warning("No new predictions to compute WER on. Reloading from output file.")
        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                references.append(row["reference_transcript"])
                predictions.append(row["predicted_transcript"])

    if not references:
        raise RuntimeError("No predictions available to compute WER.")

    score = wer(references, predictions)
    LOGGER.info("Computed WER: %.4f", score)
    return score


def main() -> None:
    args = parse_args()
    configure_logging()

    LOGGER.info("Loading Whisper model '%s' on %s", args.model, args.device)
    processed_rows = process_corpus(args)

    if args.compute_wer:
        LOGGER.info("Computing WER over processed predictions")
        score = compute_wer(processed_rows, args.output)
        print(json.dumps({"wer": score}))


if __name__ == "__main__":
    main()
