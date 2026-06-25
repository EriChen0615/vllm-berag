# SPDX-License-Identifier: Apache-2.0
"""Run standard concat-K RAG on the prepared NarrativeQA subsets."""

from __future__ import annotations

import argparse
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
    make_longbench_prompt,
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
) -> tuple[list[str], list[int], list[int]]:
    prompts = []
    raw_prompt_token_lengths = []
    effective_prompt_token_lengths = []
    for row in rows:
        context = make_standard_rag_context(row["chunks"])
        user_prompt = make_longbench_prompt(context, row["question"])
        rendered_prompt = render_qwen_chat_prompt(
            tokenizer,
            user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        prompts.append(rendered_prompt)
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
    prompts: list[str],
    raw_prompt_token_lengths: list[int],
    effective_prompt_token_lengths: list[int],
    preview_count: int,
) -> None:
    for index, (row, prompt, raw_tokens, effective_tokens) in enumerate(
        zip(rows, prompts, raw_prompt_token_lengths, effective_prompt_token_lengths)
    ):
        if index >= preview_count:
            break
        has_qwen_markers = "<|im_start|>" in prompt or "<|im_end|>" in prompt
        has_longbench_prompt = "Story:" in prompt and "Question:" in prompt
        print(
            f"[prompt] k={k_value} example_id={row['example_id']} "
            f"raw_prompt_tokens={raw_tokens} "
            f"effective_prompt_tokens={effective_tokens} "
            f"qwen_markers={has_qwen_markers} "
            f"longbench_content={has_longbench_prompt}"
        )
        print(prompt[:1200])
        if len(prompt) > 1200:
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


def run_one_k(
    *,
    args: argparse.Namespace,
    llm: Any,
    sampling_params: Any,
    k_value: int,
    rows: list[dict[str, Any]],
    prompts: list[str],
    raw_prompt_token_lengths: list[int],
    effective_prompt_token_lengths: list[int],
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
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            tokenization_kwargs=make_tokenization_kwargs(args),
            use_tqdm=not args.disable_tqdm,
        )
        wall_time_s = time.perf_counter() - start
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
    metrics.update({"status": "ok", "k": k_value})

    write_jsonl(k_dir / "predictions.jsonl", prediction_rows)
    write_json(k_dir / "metrics.json", metrics)
    print(
        f"[rag] k={k_value} requests={len(rows)} wall_time={wall_time_s:.2f}s "
        f"rps={metrics['requests_per_second']:.4f} "
        f"mean_input_tokens={metrics['mean_input_tokens']:.1f} "
        f"p90_ttft={metrics['p90_ttft_s']:.4f}s "
        f"p90_tpot={metrics['p90_tpot_s']:.4f}s "
        f"bleu={metrics['corpus_bleu']:.4f}"
    )


def main() -> None:
    set_cache_env()
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args)
    rendered_by_k: dict[
        int,
        tuple[list[dict[str, Any]], list[str], list[int], list[int]],
    ] = {}
    for k_value in k_values:
        rows = load_k_rows(data_dir, k_value, args.max_examples)
        prompts, raw_prompt_token_lengths, effective_prompt_token_lengths = (
            render_prompts(rows, tokenizer, args)
        )
        rendered_by_k[k_value] = (
            rows,
            prompts,
            raw_prompt_token_lengths,
            effective_prompt_token_lengths,
        )
        print(
            f"[rag] prepared k={k_value} rows={len(rows)} "
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
        )


if __name__ == "__main__":
    main()
