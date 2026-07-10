# SPDX-License-Identifier: Apache-2.0
"""Run standard concat-K RAG on the prepared NarrativeQA subsets."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_scripts.narrativeqa_benchmark_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_SYSTEM_PROMPT,
    aggregate_prediction_metrics,
    build_prediction_row,
    length_summary,
    make_narrativeqa_prompt,
    make_standard_rag_context,
    parse_k_values,
    read_jsonl,
    render_qwen_chat_prompt,
    write_json,
    write_jsonl,
)


CACHE_ROOT = REPO_ROOT / ".cache"
TMP_ROOT = REPO_ROOT / ".tmp"


def set_cache_env() -> None:
    env_defaults = {
        "TMPDIR": TMP_ROOT,
        "TEMP": TMP_ROOT,
        "TMP": TMP_ROOT,
        "XDG_CACHE_HOME": CACHE_ROOT,
        "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
        "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
        "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT / "torchinductor",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, str(value))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--k-values",
        default=",".join(str(k_value) for k_value in DEFAULT_K_VALUES),
    )
    parser.add_argument("--data-dir", default="my_outputs/data/NarrativeQA")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=None,
        help=(
            "Number of standard RAG requests per llm.generate call. "
            "Use 1 to benchmark batch-size-one execution over many examples."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--truncate-prompt-tokens", type=int, default=None)
    parser.add_argument("--truncation-side", choices=("left", "right"), default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--dry-run-prompts", action="store_true")
    parser.add_argument("--preview-prompts", type=int, default=2)
    parser.add_argument("--query-image-path", default=None)
    parser.add_argument(
        "--query-image-uuid",
        default="narrativeqa-shared-query-image",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if (
        args.truncate_prompt_tokens is not None
        and args.truncate_prompt_tokens >= 0
        and args.truncate_prompt_tokens + args.max_tokens > args.max_model_len
    ):
        raise ValueError(
            "truncate_prompt_tokens + max_tokens must be <= max_model_len; "
            f"got {args.truncate_prompt_tokens} + {args.max_tokens} > "
            f"{args.max_model_len}"
        )
    return args


def load_query_image(args: argparse.Namespace) -> Any | None:
    if args.query_image_path is None:
        return None
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --query-image-path.") from exc
    image_path = Path(args.query_image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing query image: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def make_prompt_with_image(
    prompt: str,
    query_image: Any | None,
    query_image_uuid: str,
) -> str | dict[str, Any]:
    if query_image is None:
        return prompt
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": query_image},
        "multi_modal_uuids": {"image": [query_image_uuid]},
    }


def load_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to render Qwen prompts.") from exc
    return AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )


def render_prompts(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[str | dict[str, Any]], list[int], list[int]]:
    prompts = []
    raw_prompt_token_lengths = []
    effective_prompt_token_lengths = []
    query_image = load_query_image(args)
    for row in rows:
        context = make_standard_rag_context(row["chunks"])
        user_prompt = make_narrativeqa_prompt(
            context,
            row["question"],
            include_image=query_image is not None,
        )
        rendered_prompt = render_qwen_chat_prompt(
            tokenizer,
            user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            include_image=query_image is not None,
        )
        prompts.append(
            make_prompt_with_image(
                rendered_prompt,
                query_image,
                args.query_image_uuid,
            )
        )
        raw_length = len(tokenizer.encode(rendered_prompt, add_special_tokens=False))
        raw_prompt_token_lengths.append(raw_length)
        if args.truncate_prompt_tokens is None:
            effective_length = raw_length
        elif args.truncate_prompt_tokens == -1:
            effective_length = min(raw_length, args.max_model_len)
        else:
            effective_length = min(raw_length, args.truncate_prompt_tokens)
        effective_prompt_token_lengths.append(effective_length)
    return prompts, raw_prompt_token_lengths, effective_prompt_token_lengths


def load_k_rows(
    data_dir: Path,
    k_value: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    path = data_dir / f"narrativeqa_k{k_value}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared data file: {path}")
    return read_jsonl(path, limit=max_examples)


def print_prompt_preview(
    *,
    k_value: int,
    rows: list[dict[str, Any]],
    prompts: list[str | dict[str, Any]],
    raw_prompt_token_lengths: list[int],
    effective_prompt_token_lengths: list[int],
    preview_count: int,
) -> None:
    for index, (row, prompt, raw_tokens, effective_tokens) in enumerate(
        zip(rows, prompts, raw_prompt_token_lengths, effective_prompt_token_lengths)
    ):
        if index >= preview_count:
            break
        prompt_text = prompt["prompt"] if isinstance(prompt, dict) else prompt
        has_qwen_markers = (
            "<|im_start|>" in prompt_text or "<|im_end|>" in prompt_text
        )
        has_image_prompt = isinstance(prompt, dict) and "multi_modal_data" in prompt
        has_longbench_prompt = "Story:" in prompt_text and "Question:" in prompt_text
        print(
            f"[prompt] k={k_value} example_id={row['example_id']} "
            f"raw_prompt_tokens={raw_tokens} "
            f"effective_prompt_tokens={effective_tokens} "
            f"qwen_markers={has_qwen_markers} "
            f"image_prompt={has_image_prompt} "
            f"longbench_content={has_longbench_prompt}"
        )
        print(prompt_text[:1200])
        if len(prompt_text) > 1200:
            print("[prompt] ...")


def make_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": False,
        "disable_log_stats": False,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    return LLM(**llm_kwargs)


def make_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )


def make_tokenization_kwargs(args: argparse.Namespace) -> dict[str, Any] | None:
    tokenization_kwargs: dict[str, Any] = {}
    if args.truncate_prompt_tokens is not None:
        tokenization_kwargs["truncate_prompt_tokens"] = args.truncate_prompt_tokens
    if args.truncation_side is not None:
        tokenization_kwargs["truncation_side"] = args.truncation_side
    return tokenization_kwargs or None


def write_run_config(
    *,
    output_path: Path,
    args: argparse.Namespace,
    k_value: int,
    num_examples: int,
    raw_prompt_token_lengths: list[int],
    effective_prompt_token_lengths: list[int],
) -> None:
    write_json(
        output_path,
        {
            "model": args.model,
            "k": k_value,
            "num_examples": num_examples,
            "max_examples": args.max_examples,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "truncate_prompt_tokens": args.truncate_prompt_tokens,
            "truncation_side": args.truncation_side,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "request_batch_size": args.request_batch_size,
            "query_image_path": args.query_image_path,
            "query_image_uuid": args.query_image_uuid,
            "raw_prompt_token_summary": length_summary(raw_prompt_token_lengths),
            "prompt_token_summary": length_summary(effective_prompt_token_lengths),
        },
    )


def is_oom_like(exc: BaseException) -> bool:
    error_text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in error_text
        for marker in (
            "out of memory",
            "cuda error",
            "cublas",
            "cudnn",
            "allocation",
            "allocate",
        )
    )


def write_failed_metrics(
    *,
    k_dir: Path,
    k_value: int,
    rows: list[dict[str, Any]],
    wall_time_s: float,
    exc: BaseException,
) -> None:
    write_jsonl(k_dir / "predictions.jsonl", [])
    write_json(
        k_dir / "metrics.json",
        {
            "status": "failed",
            "k": k_value,
            "num_requests": len(rows),
            "wall_time_s": wall_time_s,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _merge_scheduler_stats(
    totals: dict[str, float],
    scheduler_stats: dict[str, Any],
) -> None:
    for key, value in scheduler_stats.items():
        if not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if key in {
            "gpu_kv_cache_usage_peak",
            "gpu_kv_cache_usage_peak_pct",
        }:
            totals[key] = max(totals.get(key, 0.0), numeric_value)
        elif key in {
            "gpu_kv_cache_usage_final",
            "gpu_kv_cache_usage_final_pct",
        }:
            totals[key] = numeric_value
        elif key.endswith("_hit_rate") or key.endswith("_hit_rate_pct"):
            continue
        else:
            totals[key] = totals.get(key, 0.0) + numeric_value


def _finalize_scheduler_stats(totals: dict[str, float]) -> dict[str, float]:
    stats = dict(totals)
    prefix_queries = stats.get("prefix_cache_queries", 0.0)
    prefix_hits = stats.get("prefix_cache_hits", 0.0)
    connector_queries = stats.get("connector_prefix_cache_queries", 0.0)
    connector_hits = stats.get("connector_prefix_cache_hits", 0.0)
    stats["prefix_cache_hit_rate"] = (
        prefix_hits / prefix_queries if prefix_queries > 0 else 0.0
    )
    stats["prefix_cache_hit_rate_pct"] = stats["prefix_cache_hit_rate"] * 100
    stats["connector_prefix_cache_hit_rate"] = (
        connector_hits / connector_queries if connector_queries > 0 else 0.0
    )
    stats["connector_prefix_cache_hit_rate_pct"] = (
        stats["connector_prefix_cache_hit_rate"] * 100
    )
    return stats


def _merge_generate_timing(
    totals: dict[str, float],
    scheduler_totals: dict[str, float],
    timing: dict[str, Any],
) -> None:
    for key in ("total_s", "add_requests_s", "run_engine_s"):
        value = timing.get(key)
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0.0) + float(value)
    scheduler_stats = timing.get("scheduler_stats") or {}
    if isinstance(scheduler_stats, dict):
        _merge_scheduler_stats(scheduler_totals, scheduler_stats)


def run_one_k(
    *,
    args: argparse.Namespace,
    llm: Any,
    sampling_params: Any,
    k_value: int,
    rows: list[dict[str, Any]],
    prompts: list[str | dict[str, Any]],
    raw_prompt_token_lengths: list[int],
    effective_prompt_token_lengths: list[int],
    prompt_render_s: float,
) -> None:
    k_dir = Path(args.output_dir) / "standard_rag" / f"k{k_value}"
    k_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        output_path=k_dir / "run_config.json",
        args=args,
        k_value=k_value,
        num_examples=len(rows),
        raw_prompt_token_lengths=raw_prompt_token_lengths,
        effective_prompt_token_lengths=effective_prompt_token_lengths,
    )

    start = time.perf_counter()
    try:
        outputs = []
        timing_totals: dict[str, float] = {}
        scheduler_totals: dict[str, float] = {}
        batch_size = args.request_batch_size or len(prompts)
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            outputs.extend(
                llm.generate(
                    prompts[batch_start:batch_end],
                    sampling_params=sampling_params,
                    tokenization_kwargs=make_tokenization_kwargs(args),
                    use_tqdm=not args.disable_tqdm,
                )
            )
            _merge_generate_timing(
                timing_totals,
                scheduler_totals,
                getattr(llm, "_last_generate_timing", {}),
            )
        wall_time_s = time.perf_counter() - start
        if len(outputs) != len(rows):
            raise RuntimeError(
                f"Expected {len(rows)} standard RAG outputs, got {len(outputs)}."
            )
    except Exception as exc:
        wall_time_s = time.perf_counter() - start
        write_failed_metrics(
            k_dir=k_dir,
            k_value=k_value,
            rows=rows,
            wall_time_s=wall_time_s,
            exc=exc,
        )
        if args.stop_on_error or not is_oom_like(exc):
            raise
        print(f"[rag] k={k_value} failed with OOM-like error; metrics saved")
        return

    prediction_rows = [
        build_prediction_row(
            example=row,
            output=output,
            prompt_tokens=effective_prompt_token_lengths[index],
        )
        for index, (row, output) in enumerate(zip(rows, outputs))
    ]
    metrics = aggregate_prediction_metrics(prediction_rows, wall_time_s=wall_time_s)
    scheduler_stats = _finalize_scheduler_stats(scheduler_totals)
    metrics.update(
        {
            "status": "ok",
            "k": k_value,
            "prompt_render_s": prompt_render_s,
            "request_batch_size": args.request_batch_size,
            "num_request_batches": math.ceil(len(rows) / batch_size),
            "generate_total_s": timing_totals.get("total_s", wall_time_s),
            "generate_add_requests_s": timing_totals.get("add_requests_s"),
            "generate_run_engine_s": timing_totals.get("run_engine_s"),
            **scheduler_stats,
        }
    )

    write_jsonl(k_dir / "predictions.jsonl", prediction_rows)
    write_json(k_dir / "metrics.json", metrics)
    add_requests_s = metrics["generate_add_requests_s"] or 0.0
    run_engine_s = metrics["generate_run_engine_s"] or 0.0
    kv_peak_pct = metrics.get("gpu_kv_cache_usage_peak_pct", 0.0)
    prefix_hit_pct = metrics.get("prefix_cache_hit_rate_pct", 0.0)
    print(
        f"[rag] k={k_value} requests={len(rows)} wall_time={wall_time_s:.2f}s "
        f"render={prompt_render_s:.2f}s "
        f"add_requests={add_requests_s:.2f}s "
        f"run_engine={run_engine_s:.2f}s "
        f"kv_peak={kv_peak_pct:.1f}% "
        f"prefix_hit={prefix_hit_pct:.1f}% "
        f"rps={metrics['requests_per_second']:.4f} "
        f"mean_input_tokens={metrics['mean_input_tokens']:.1f} "
        f"p90_ttft={metrics['p90_ttft_s']:.4f}s "
        f"p90_tpot={metrics['p90_tpot_s']:.4f}s "
        f"bleu={metrics['corpus_bleu']:.4f}"
    )


def main() -> None:
    set_cache_env()
    args = parse_args()
    if args.request_batch_size is not None and args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive when set.")
    k_values = parse_k_values(args.k_values)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args)
    rendered_by_k: dict[
        int,
        tuple[
            list[dict[str, Any]],
            list[str | dict[str, Any]],
            list[int],
            list[int],
            float,
        ],
    ] = {}
    for k_value in k_values:
        rows = load_k_rows(data_dir, k_value, args.max_examples)
        render_start = time.perf_counter()
        prompts, raw_prompt_token_lengths, effective_prompt_token_lengths = (
            render_prompts(rows, tokenizer, args)
        )
        render_s = time.perf_counter() - render_start
        rendered_by_k[k_value] = (
            rows,
            prompts,
            raw_prompt_token_lengths,
            effective_prompt_token_lengths,
            render_s,
        )
        print(
            f"[rag] prepared k={k_value} rows={len(rows)} "
            f"render_s={render_s:.2f}s "
            f"prompt_tokens={length_summary(effective_prompt_token_lengths)} "
            f"raw_prompt_tokens={length_summary(raw_prompt_token_lengths)}"
        )
        if args.dry_run_prompts:
            print_prompt_preview(
                k_value=k_value,
                rows=rows,
                prompts=prompts,
                raw_prompt_token_lengths=raw_prompt_token_lengths,
                effective_prompt_token_lengths=effective_prompt_token_lengths,
                preview_count=args.preview_prompts,
            )

    if args.dry_run_prompts:
        return

    llm = make_llm(args)
    sampling_params = make_sampling_params(args)
    for k_value in k_values:
        (
            rows,
            prompts,
            raw_prompt_token_lengths,
            effective_prompt_token_lengths,
            prompt_render_s,
        ) = rendered_by_k[k_value]
        run_one_k(
            args=args,
            llm=llm,
            sampling_params=sampling_params,
            k_value=k_value,
            rows=rows,
            prompts=prompts,
            raw_prompt_token_lengths=raw_prompt_token_lengths,
            effective_prompt_token_lengths=effective_prompt_token_lengths,
            prompt_render_s=prompt_render_s,
        )


if __name__ == "__main__":
    main()
