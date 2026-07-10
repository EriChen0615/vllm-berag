# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the NarrativeQA RAG benchmark scripts."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_K_VALUES = (50, 75, 100, 150, 200)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
LONG_BENCH_NARRATIVEQA_PROMPT = (
    "You are given a story, which can be either a novel or a movie script, "
    "and a question.\n"
    "Answer the question as concisely as you can, using a single phrase if "
    "possible.\n"
    "Answer in at most one sentence.\n"
    "Do not provide any explanation.\n\n"
    "Story: {context}\n\n"
    "Now, answer the question based on the story as concisely as you can, "
    "using a single phrase if possible.\n"
    "Answer in at most one sentence.\n"
    "Do not provide any explanation.\n\n"
    "Question: {input}\n\n"
    "Answer:"
)
IMAGE_NARRATIVEQA_PROMPT = (
    "You are given an image, a story, and a question.\n"
    "First, describe the image in one concise sentence.\n"
    "Then answer the question based on the story as concisely as you can, "
    "using a single phrase if possible.\n"
    "Answer in at most one sentence.\n\n"
    "Response format:\n"
    "Image: <one-sentence image description>\n"
    "Answer: <short answer>\n\n"
    "Story: {context}\n\n"
    "Question: {input}\n\n"
    "Image:"
)

DOCUMENT_ID_FIELDS = ("chunk_id", "id", "_id", "document_id")
DOCUMENT_TEXT_FIELDS = ("text", "content", "document", "chunk", "passage", "context")
QUERY_ID_FIELDS = ("id", "_id", "query_id")
QUERY_TEXT_FIELDS = ("og_query", "query", "question")
ANSWER_FIELDS = ("answer", "answers")
GOLD_CHUNK_FIELDS = ("chunk_id", "gold_chunk_id", "document_id")

_TRAILING_CHUNK_RE = re.compile(r"^(?P<base>.+)_(?P<index>\d+)$")


def parse_k_values(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
        parsed = [int(part) for part in raw_values]
    else:
        parsed = [int(part) for part in value]

    if not parsed:
        raise ValueError("At least one K value is required.")
    if min(parsed) <= 0:
        raise ValueError("K values must be positive.")

    seen: set[int] = set()
    k_values = []
    for k_value in parsed:
        if k_value not in seen:
            k_values.append(k_value)
            seen.add(k_value)
    return k_values


def coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [coerce_to_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def first_text_field(row: Mapping[str, Any], field_names: Sequence[str]) -> str:
    for field_name in field_names:
        if field_name in row:
            value = coerce_to_text(row[field_name])
            if value:
                return value
    return ""


def stable_hash_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def base_doc_id(chunk_id: str) -> str:
    match = _TRAILING_CHUNK_RE.match(str(chunk_id))
    return match.group("base") if match else str(chunk_id)


def chunk_sort_key(chunk_id: str) -> tuple[str, int, str]:
    chunk_id = str(chunk_id)
    match = _TRAILING_CHUNK_RE.match(chunk_id)
    if match:
        return (match.group("base"), int(match.group("index")), chunk_id)
    return (chunk_id, -1, chunk_id)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def build_nested_chunk_ids(
    *,
    example_id: str,
    gold_chunk_id: str,
    same_document_chunk_ids: Sequence[str],
    all_chunk_ids: Sequence[str],
    k_values: Sequence[int],
    seed: int,
    gold_window: int = 50,
) -> tuple[list[str], int] | None:
    """Build one max-K chunk list whose prefixes form all smaller K sets."""

    k_values = parse_k_values(k_values)
    max_k = max(k_values)
    min_k = min(k_values)
    gold_chunk_id = str(gold_chunk_id)
    all_chunk_ids = unique_preserve_order(str(chunk_id) for chunk_id in all_chunk_ids)

    if gold_chunk_id not in set(all_chunk_ids):
        return None

    rng_key = f"{seed}:{example_id}:{gold_chunk_id}"
    rng = random.Random(stable_hash_int(rng_key))

    same_doc = [
        str(chunk_id)
        for chunk_id in same_document_chunk_ids
        if str(chunk_id) != gold_chunk_id
    ]
    same_doc = sorted(unique_preserve_order(same_doc), key=chunk_sort_key)

    excluded = set(same_doc)
    excluded.add(gold_chunk_id)
    distractors = [chunk_id for chunk_id in all_chunk_ids if chunk_id not in excluded]
    rng.shuffle(distractors)

    pool = same_doc + distractors
    if len(pool) + 1 < max_k:
        return None

    gold_position = rng.randrange(min(gold_window, min_k))
    selected = pool[:gold_position] + [gold_chunk_id] + pool[gold_position:]
    selected = selected[:max_k]

    if len(selected) != max_k or gold_chunk_id not in selected[:min_k]:
        return None
    return selected, gold_position


def make_standard_rag_context(chunks: Sequence[str]) -> str:
    return "\n\n".join(
        f"[Chunk {index + 1}]\n{chunk_text}"
        for index, chunk_text in enumerate(chunks)
    )


def make_longbench_prompt(context: str, question: str) -> str:
    return LONG_BENCH_NARRATIVEQA_PROMPT.format(context=context, input=question)


def make_narrativeqa_prompt(
    context: str,
    question: str,
    *,
    include_image: bool = False,
) -> str:
    template = IMAGE_NARRATIVEQA_PROMPT if include_image else (
        LONG_BENCH_NARRATIVEQA_PROMPT
    )
    return template.format(context=context, input=question)


def render_qwen_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_image: bool = False,
) -> str:
    user_content: str | list[dict[str, str]]
    if include_image:
        user_content = [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def percentile(values: Sequence[float], p: float) -> float:
    clean_values = sorted(float(value) for value in values if value is not None)
    if not clean_values:
        return 0.0
    if len(clean_values) == 1:
        return clean_values[0]

    rank = (len(clean_values) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean_values[low]
    low_weight = high - rank
    high_weight = rank - low
    return clean_values[low] * low_weight + clean_values[high] * high_weight


def mean(values: Sequence[float]) -> float:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return 0.0
    return sum(clean_values) / len(clean_values)


def length_summary(values: Sequence[int]) -> dict[str, float | int]:
    clean_values = [int(value) for value in values]
    if not clean_values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "count": len(clean_values),
        "min": min(clean_values),
        "max": max(clean_values),
        "mean": mean(clean_values),
        "p50": percentile(clean_values, 50),
        "p90": percentile(clean_values, 90),
    }


def get_request_timing_metrics(output: Any, output_tokens: int) -> dict[str, float]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        raise RuntimeError(
            "RequestOutput.metrics is None; set disable_log_stats=False."
        )

    def interval(end: float, start: float) -> float:
        if not end or not start:
            return 0.0
        return max(0.0, float(end) - float(start))

    ttft_s = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    queued_s = interval(
        getattr(metrics, "scheduled_ts", 0.0),
        getattr(metrics, "queued_ts", 0.0),
    )
    prefill_s = interval(
        getattr(metrics, "first_token_ts", 0.0),
        getattr(metrics, "scheduled_ts", 0.0),
    )
    decode_s = interval(
        getattr(metrics, "last_token_ts", 0.0),
        getattr(metrics, "first_token_ts", 0.0),
    )
    tpot_s = decode_s / (output_tokens - 1) if output_tokens > 1 else 0.0
    return {
        "ttft_s": ttft_s,
        "queued_s": queued_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tpot_s": tpot_s,
    }


def build_prediction_row(
    *,
    example: Mapping[str, Any],
    output: Any,
    prompt_tokens: int | None = None,
) -> dict[str, Any]:
    completion = output.outputs[0]
    output_token_ids = [int(token_id) for token_id in completion.token_ids]
    output_tokens = len(output_token_ids)
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        prompt_tokens = len(prompt_token_ids)

    timing = get_request_timing_metrics(output, output_tokens)
    return {
        "example_id": example["example_id"],
        "k": int(example["k"]),
        "question": example["question"],
        "reference": example["answer"],
        "prediction": completion.text,
        "output_token_ids": output_token_ids,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": timing["ttft_s"],
        "queued_s": timing["queued_s"],
        "prefill_s": timing["prefill_s"],
        "decode_s": timing["decode_s"],
        "tpot_s": timing["tpot_s"],
        "finished": bool(output.finished),
        "finish_reason": completion.finish_reason,
    }


def compute_corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "sacrebleu is required. Run: source my_scripts/activate_env.sh && "
            "uv pip install sacrebleu"
        ) from exc
    return float(sacrebleu.corpus_bleu(list(predictions), [list(references)]).score)


def aggregate_prediction_metrics(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    wall_time_s: float,
) -> dict[str, Any]:
    num_requests = len(prediction_rows)
    predictions = [str(row["prediction"]) for row in prediction_rows]
    references = [str(row["reference"]) for row in prediction_rows]

    metrics = {
        "num_requests": num_requests,
        "wall_time_s": wall_time_s,
        "requests_per_second": num_requests / wall_time_s if wall_time_s > 0 else 0.0,
        "mean_input_tokens": mean([row["prompt_tokens"] for row in prediction_rows]),
        "mean_output_tokens": mean([row["output_tokens"] for row in prediction_rows]),
        "p50_ttft_s": percentile([row["ttft_s"] for row in prediction_rows], 50),
        "p90_ttft_s": percentile([row["ttft_s"] for row in prediction_rows], 90),
        "p50_prefill_s": percentile([row["prefill_s"] for row in prediction_rows], 50),
        "p90_prefill_s": percentile([row["prefill_s"] for row in prediction_rows], 90),
        "p50_tpot_s": percentile([row["tpot_s"] for row in prediction_rows], 50),
        "p90_tpot_s": percentile([row["tpot_s"] for row in prediction_rows], 90),
        "p50_decode_s": percentile([row["decode_s"] for row in prediction_rows], 50),
        "p90_decode_s": percentile([row["decode_s"] for row in prediction_rows], 90),
        "corpus_bleu": compute_corpus_bleu(predictions, references),
    }
    return metrics
