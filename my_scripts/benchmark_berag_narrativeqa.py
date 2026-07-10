# SPDX-License-Identifier: Apache-2.0
"""Run BERAG smoke experiments on prepared NarrativeQA subsets."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_scripts.narrativeqa_benchmark_utils import (  # noqa: E402
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
BERAG_CONTEXT_SENTINEL = "__BERAG_CONTEXT_SENTINEL__"


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
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--k-values", default="3,8")
    parser.add_argument("--data-dir", default="my_outputs/data/NarrativeQA")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=None,
        help=(
            "Number of parent BERAG requests per generate_berag call. "
            "Use 1 to benchmark batch-size-one execution over many examples."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--num-accumulator-rows", type=int, default=400)
    parser.add_argument("--pruning-top-p", type=float, default=1.0)
    parser.add_argument(
        "--prior-mode",
        choices=("uniform", "module"),
        default="uniform",
    )
    parser.add_argument(
        "--prior-module-cls",
        default="tests.berag.prior_fixtures.TinyPrior",
    )
    parser.add_argument("--prior-module-weights-path", default=None)
    parser.add_argument("--prior-hidden-size", type=int, default=None)
    parser.add_argument("--default-prior-token-offset", type=int, default=-4)
    parser.add_argument("--berag-log-groups", action="store_true")
    parser.add_argument("--berag-log-full-posterior", action="store_true")
    parser.add_argument("--berag-group-trace-path", default=None)
    parser.add_argument("--query-image-path", default=None)
    parser.add_argument("--query-image-uuid", default="narrativeqa-shared-query-image")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to render Qwen prompts.") from exc
    return AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )


def resolve_hidden_size(args: argparse.Namespace) -> int:
    if args.prior_hidden_size is not None:
        return args.prior_hidden_size
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to infer hidden size.") from exc
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not infer hidden_size from model config: {args.model}")
    return int(hidden_size)


def ensure_prior_weights(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.prior_mode == "uniform":
        return None
    if args.prior_module_weights_path:
        return Path(args.prior_module_weights_path)

    import torch

    from tests.berag.prior_fixtures import TinyPrior

    hidden_size = resolve_hidden_size(args)
    prior = TinyPrior(hidden_size=hidden_size)
    prior_dir = output_dir / "berag_prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_path = prior_dir / f"tiny_prior_h{hidden_size}.pt"
    if not prior_path.exists():
        torch.save(prior.state_dict(), prior_path)
    return prior_path


def make_llm(args: argparse.Namespace, prior_path: Path | None) -> Any:
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
        "berag_num_accumulator_rows": args.num_accumulator_rows,
        "berag_prior_mode": args.prior_mode,
        "berag_default_prior_token_offset": args.default_prior_token_offset,
    }
    if args.prior_mode == "module":
        hidden_size = resolve_hidden_size(args)
        assert prior_path is not None
        llm_kwargs.update(
            {
                "berag_prior_module_cls": args.prior_module_cls,
                "berag_prior_module_weights_path": str(prior_path),
                "berag_prior_module_kwargs": {"hidden_size": hidden_size},
            }
        )
    if args.berag_log_groups:
        if args.berag_group_trace_path:
            trace_path = Path(args.berag_group_trace_path)
        else:
            trace_path = Path(args.output_dir) / "berag" / "group_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        args.resolved_berag_group_trace_path = str(trace_path)
        llm_kwargs.update(
            {
                "berag_group_trace_path": str(trace_path),
                "berag_group_trace_full_posterior": (
                    args.berag_log_full_posterior
                ),
            }
        )
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
        ignore_eos=False,
    )


def make_berag_params(args: argparse.Namespace) -> Any:
    from vllm.berag import BeragParams

    return BeragParams(pruning_top_p=args.pruning_top_p)


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


def make_shared_prefix_prompt(
    shared_prefix: str,
    query_image: Any | None,
    query_image_uuid: str,
) -> str | dict[str, Any]:
    if query_image is None:
        return shared_prefix
    return {
        "prompt": shared_prefix,
        "multi_modal_data": {"image": query_image},
        "multi_modal_uuids": {"image": [query_image_uuid]},
    }


def split_berag_prompt(
    tokenizer: Any,
    question: str,
    *,
    include_image: bool = False,
) -> tuple[str, str]:
    user_prompt = make_narrativeqa_prompt(
        BERAG_CONTEXT_SENTINEL,
        question,
        include_image=include_image,
    )
    rendered = render_qwen_chat_prompt(
        tokenizer,
        user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        include_image=include_image,
    )
    parts = rendered.split(BERAG_CONTEXT_SENTINEL)
    if len(parts) != 2:
        raise ValueError("BERAG context sentinel was not rendered exactly once.")
    return parts[0], parts[1]


def make_branch_documents(chunks: list[str]) -> list[str]:
    return [
        f"[Chunk {index + 1}]\n{chunk_text}"
        for index, chunk_text in enumerate(chunks)
    ]


def logical_prompt_tokens(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    include_image: bool = False,
) -> int:
    context = make_standard_rag_context(row["chunks"])
    user_prompt = make_narrativeqa_prompt(
        context,
        row["question"],
        include_image=include_image,
    )
    rendered = render_qwen_chat_prompt(
        tokenizer,
        user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        include_image=include_image,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def branch_prompt_token_lengths(
    tokenizer: Any,
    shared_prefix: str,
    documents: list[str],
    suffix: str,
) -> list[int]:
    return [
        len(
            tokenizer.encode(
                f"{shared_prefix}{document}{suffix}",
                add_special_tokens=False,
            )
        )
        for document in documents
    ]


def load_k_rows(
    data_dir: Path,
    k_value: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    path = data_dir / f"narrativeqa_k{k_value}.jsonl"
    if path.exists():
        return read_jsonl(path, limit=max_examples)

    available: list[tuple[int, Path]] = []
    for candidate in data_dir.glob("narrativeqa_k*.jsonl"):
        match = re.fullmatch(r"narrativeqa_k(\d+)\.jsonl", candidate.name)
        if match:
            available.append((int(match.group(1)), candidate))
    for source_k, source_path in sorted(available):
        if source_k < k_value:
            continue
        rows = read_jsonl(source_path, limit=max_examples)
        derived_rows = []
        for row in rows:
            derived = dict(row)
            derived["k"] = k_value
            derived["chunk_ids"] = row["chunk_ids"][:k_value]
            derived["chunks"] = row["chunks"][:k_value]
            derived_rows.append(derived)
        print(
            f"[berag] derived k={k_value} from {source_path.name} "
            f"rows={len(derived_rows)}"
        )
        return derived_rows

    raise FileNotFoundError(f"Missing prepared data file: {path}")


def write_run_config(
    *,
    output_path: Path,
    args: argparse.Namespace,
    k_value: int,
    rows: list[dict[str, Any]],
    logical_prompt_lengths: list[int],
    branch_prompt_totals: list[int],
) -> None:
    write_json(
        output_path,
        {
            "model": args.model,
            "k": k_value,
            "num_examples": len(rows),
            "max_examples": args.max_examples,
            "request_batch_size": args.request_batch_size,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "num_accumulator_rows": args.num_accumulator_rows,
            "pruning_top_p": args.pruning_top_p,
            "prior_mode": args.prior_mode,
            "prior_module_cls": args.prior_module_cls,
            "prior_module_weights_path": getattr(
                args,
                "resolved_prior_module_weights_path",
                args.prior_module_weights_path,
            ),
            "default_prior_token_offset": args.default_prior_token_offset,
            "query_image_path": args.query_image_path,
            "query_image_uuid": args.query_image_uuid,
            "berag_group_trace_path": getattr(
                args,
                "resolved_berag_group_trace_path",
                args.berag_group_trace_path,
            ),
            "berag_log_full_posterior": args.berag_log_full_posterior,
            "prompt_token_summary": length_summary(logical_prompt_lengths),
            "branch_prompt_tokens_total_summary": length_summary(
                branch_prompt_totals
            ),
        },
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


def _merge_berag_timing(
    totals: dict[str, float],
    scheduler_totals: dict[str, float],
    timing: dict[str, Any],
) -> None:
    for key in (
        "total_s",
        "admission_s",
        "run_engine_s",
        "num_parent_requests",
        "num_child_requests",
    ):
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
    tokenizer: Any,
    sampling_params: Any,
    berag_params: Any,
    query_image: Any | None,
    k_value: int,
    rows: list[dict[str, Any]],
) -> None:
    k_dir = Path(args.output_dir) / "berag" / f"k{k_value}"
    k_dir.mkdir(parents=True, exist_ok=True)

    shared_prefixes: list[str | dict[str, Any]] = []
    document_lists: list[list[str]] = []
    suffixes: list[str] = []
    request_ids: list[str] = []
    logical_prompt_lengths: list[int] = []
    branch_prompt_totals: list[int] = []

    prepare_start = time.perf_counter()
    for row in rows:
        shared_prefix_text, suffix = split_berag_prompt(
            tokenizer,
            row["question"],
            include_image=query_image is not None,
        )
        documents = make_branch_documents(row["chunks"])
        branch_lengths = branch_prompt_token_lengths(
            tokenizer,
            shared_prefix_text,
            documents,
            suffix,
        )
        shared_prefixes.append(
            make_shared_prefix_prompt(
                shared_prefix_text,
                query_image,
                args.query_image_uuid,
            )
        )
        document_lists.append(documents)
        suffixes.append(suffix)
        request_ids.append(f"{row['example_id']}:berag")
        logical_prompt_lengths.append(
            logical_prompt_tokens(
                tokenizer,
                row,
                include_image=query_image is not None,
            )
        )
        branch_prompt_totals.append(sum(branch_lengths))
    prepare_s = time.perf_counter() - prepare_start

    write_run_config(
        output_path=k_dir / "run_config.json",
        args=args,
        k_value=k_value,
        rows=rows,
        logical_prompt_lengths=logical_prompt_lengths,
        branch_prompt_totals=branch_prompt_totals,
    )

    start = time.perf_counter()
    try:
        outputs = []
        timing_totals: dict[str, float] = {}
        scheduler_totals: dict[str, float] = {}
        batch_size = args.request_batch_size or len(rows)
        for batch_start in range(0, len(rows), batch_size):
            batch_end = min(batch_start + batch_size, len(rows))
            outputs.extend(
                llm.generate_berag(
                    shared_prefix=shared_prefixes[batch_start:batch_end],
                    documents=document_lists[batch_start:batch_end],
                    suffix=suffixes[batch_start:batch_end],
                    sampling_params=sampling_params,
                    berag_params=berag_params,
                    request_id=request_ids[batch_start:batch_end],
                    debug=args.debug,
                    use_tqdm=not args.disable_tqdm,
                )
            )
            _merge_berag_timing(
                timing_totals,
                scheduler_totals,
                getattr(llm, "_last_berag_timing", {}),
            )
        wall_time_s = time.perf_counter() - start
        if len(outputs) != len(rows):
            raise RuntimeError(
                f"Expected {len(rows)} BERAG outputs, got {len(outputs)}."
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
        if args.stop_on_error:
            raise
            print(f"[berag] k={k_value} failed; metrics saved")
            return

    prediction_rows: list[dict[str, Any]] = []
    for index, (row, output) in enumerate(zip(rows, outputs)):
        try:
            if output.metrics is None:
                raise RuntimeError("BERAG parent RequestOutput.metrics is None.")
            shared_prefix_text = (
                shared_prefixes[index]["prompt"]
                if isinstance(shared_prefixes[index], dict)
                else shared_prefixes[index]
            )
            branch_lengths = branch_prompt_token_lengths(
                tokenizer,
                shared_prefix_text,
                document_lists[index],
                suffixes[index],
            )
            pred_row = build_prediction_row(
                example=row,
                output=output,
                prompt_tokens=logical_prompt_lengths[index],
            )
            pred_row["prompt_tokens"] = logical_prompt_lengths[index]
            pred_row["num_branches"] = len(document_lists[index])
            pred_row["branch_prompt_tokens_total"] = sum(branch_lengths)
            pred_row["branch_prompt_tokens_mean"] = (
                sum(branch_lengths) / len(branch_lengths) if branch_lengths else 0.0
            )
            pred_row["branch_prompt_tokens_max"] = max(branch_lengths, default=0)
            prediction_rows.append(pred_row)
        except Exception as exc:
            wall_time_s = time.perf_counter() - start
            write_failed_metrics(
                k_dir=k_dir,
                k_value=k_value,
                rows=rows,
                wall_time_s=wall_time_s,
                exc=exc,
            )
            if args.stop_on_error:
                raise
            print(f"[berag] k={k_value} failed while writing predictions")
            return

    if len(prediction_rows) != len(rows):
        write_failed_metrics(
            k_dir=k_dir,
            k_value=k_value,
            rows=rows,
            wall_time_s=wall_time_s,
            exc=RuntimeError(
                f"Expected {len(rows)} prediction rows, got "
                f"{len(prediction_rows)}."
            ),
        )
        return

    metrics = aggregate_prediction_metrics(prediction_rows, wall_time_s=wall_time_s)
    scheduler_stats = _finalize_scheduler_stats(scheduler_totals)
    metrics.update(
        {
            "status": "ok",
            "k": k_value,
            "prepare_s": prepare_s,
            "request_batch_size": args.request_batch_size,
            "num_request_batches": math.ceil(len(rows) / batch_size),
            "generate_total_s": timing_totals.get("total_s", wall_time_s),
            "berag_admission_s": timing_totals.get("admission_s"),
            "berag_run_engine_s": timing_totals.get("run_engine_s"),
            "berag_num_parent_requests": timing_totals.get(
                "num_parent_requests"
            ),
            "berag_num_child_requests": timing_totals.get("num_child_requests"),
            **scheduler_stats,
            "mean_branch_prompt_tokens_total": (
                sum(branch_prompt_totals) / len(branch_prompt_totals)
                if branch_prompt_totals
                else 0.0
            ),
        }
    )
    write_jsonl(k_dir / "predictions.jsonl", prediction_rows)
    write_json(k_dir / "metrics.json", metrics)
    admission_s = metrics["berag_admission_s"] or 0.0
    run_engine_s = metrics["berag_run_engine_s"] or 0.0
    kv_peak_pct = metrics.get("gpu_kv_cache_usage_peak_pct", 0.0)
    prefix_hit_pct = metrics.get("prefix_cache_hit_rate_pct", 0.0)
    print(
        f"[berag] k={k_value} requests={len(rows)} wall_time={wall_time_s:.2f}s "
        f"prepare={prepare_s:.2f}s "
        f"admission={admission_s:.2f}s "
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
    if args.berag_log_full_posterior:
        args.berag_log_groups = True

    tokenizer = load_tokenizer(args)
    prior_path = ensure_prior_weights(args, output_dir)
    args.resolved_prior_module_weights_path = (
        str(prior_path) if prior_path is not None else None
    )
    llm = make_llm(args, prior_path)
    sampling_params = make_sampling_params(args)
    berag_params = make_berag_params(args)
    query_image = load_query_image(args)

    for k_value in k_values:
        rows = load_k_rows(data_dir, k_value, args.max_examples)
        run_one_k(
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            berag_params=berag_params,
            query_image=query_image,
            k_value=k_value,
            rows=rows,
        )


if __name__ == "__main__":
    main()
