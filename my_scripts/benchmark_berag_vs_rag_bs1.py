# SPDX-License-Identifier: Apache-2.0
"""Run a controlled batch-size-one NarrativeQA BERAG vs standard RAG benchmark."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_IMAGE_UUID = "narrativeqa-shared-query-image"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_QUERY_IMAGE_PATH = "my_data/just-a-random-picture.webp"
DEFAULT_K_VALUES = "50,100,150,200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="my_outputs/data/NarrativeQA")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--request-batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=40000)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--query-image-path", default=DEFAULT_QUERY_IMAGE_PATH)
    parser.add_argument("--query-image-uuid", default=DEFAULT_QUERY_IMAGE_UUID)
    parser.add_argument("--num-accumulator-rows", type=int, default=512)
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rag-truncate-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--rag-truncation-side",
        choices=("left", "right"),
        default="right",
    )
    parser.add_argument("--no-rag-truncation", action="store_true")
    parser.add_argument("--summary-skip-first-rows", type=int, default=1)
    parser.add_argument(
        "--run-order",
        choices=("rag-first", "berag-first"),
        default="rag-first",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def model_slug(model: str) -> str:
    slug = model.replace("/", "_").replace(":", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in slug)


def parse_k_values(k_values: str) -> list[int]:
    values = [value.strip() for value in k_values.split(",")]
    return [int(value) for value in values if value]


def add_value(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def add_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def add_common_engine_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend([
        "--model",
        args.model,
        "--data-dir",
        args.data_dir,
        "--output-dir",
        args.output_dir,
        "--k-values",
        args.k_values,
        "--max-examples",
        str(args.max_examples),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--request-batch-size",
        str(args.request_batch_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
    ])
    add_value(cmd, "--max-num-seqs", args.max_num_seqs)
    add_value(cmd, "--max-num-batched-tokens", args.max_num_batched_tokens)
    add_flag(cmd, "--enforce-eager", args.enforce_eager)
    add_flag(cmd, "--trust-remote-code", args.trust_remote_code)
    add_flag(cmd, "--disable-tqdm", args.disable_tqdm)
    add_flag(cmd, "--stop-on-error", args.stop_on_error)
    if args.query_image_path:
        cmd.extend([
            "--query-image-path",
            args.query_image_path,
            "--query-image-uuid",
            args.query_image_uuid,
        ])


def build_validate_cmd(args: argparse.Namespace) -> list[str]:
    return [
        python_executable(),
        "my_scripts/validate_narrativeqa_data.py",
        "--data-dir",
        args.data_dir,
        "--k-values",
        args.k_values,
        "--max-examples",
        str(args.max_examples),
    ]


def build_rag_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        python_executable(),
        "my_scripts/benchmark_standard_rag_narrativeqa.py",
    ]
    add_common_engine_args(cmd, args)
    if not args.no_rag_truncation:
        truncate_tokens = args.rag_truncate_prompt_tokens
        if truncate_tokens is None:
            truncate_tokens = args.max_model_len - args.max_tokens
        cmd.extend([
            "--truncate-prompt-tokens",
            str(truncate_tokens),
            "--truncation-side",
            args.rag_truncation_side,
        ])
    return cmd


def build_berag_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        python_executable(),
        "my_scripts/benchmark_berag_narrativeqa.py",
    ]
    add_common_engine_args(cmd, args)
    cmd.extend([
        "--num-accumulator-rows",
        str(args.num_accumulator_rows),
        "--pruning-top-p",
        str(args.pruning_top_p),
        "--prior-mode",
        args.prior_mode,
        "--default-prior-token-offset",
        str(args.default_prior_token_offset),
    ])
    if args.prior_mode == "module":
        add_value(cmd, "--prior-module-cls", args.prior_module_cls)
        add_value(cmd, "--prior-module-weights-path",
                  args.prior_module_weights_path)
        add_value(cmd, "--prior-hidden-size", args.prior_hidden_size)
    if args.berag_log_groups:
        cmd.append("--berag-log-groups")
        cmd.extend([
            "--berag-group-trace-path",
            str(Path(args.output_dir) / "berag" / "group_trace.jsonl"),
        ])
    add_flag(cmd, "--berag-log-full-posterior", args.berag_log_full_posterior)
    add_flag(cmd, "--debug", args.debug)
    return cmd


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile_value / 100
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def method_label(method_name: str) -> str:
    return "RAG" if method_name == "standard_rag" else "BERAG"


def metric_pair_cell(record: dict[str, object] | None, percentile_name: str) -> str:
    if record is None or record.get("status") != "ok":
        return "failed"
    ttft_ms = record.get(f"{percentile_name}_ttft_ms")
    tpot_ms = record.get(f"{percentile_name}_tpot_ms")
    if not isinstance(ttft_ms, (int, float)) or not isinstance(
        tpot_ms, (int, float)
    ):
        return "n/a"
    return f"{ttft_ms:.1f}, {tpot_ms:.1f}"


def build_markdown_table(
    *,
    rows_by_method: dict[str, dict[int, dict[str, object]]],
    k_values: list[int],
    percentile_name: str,
) -> str:
    header = ["Method", *[f"K = {k_value}" for k_value in k_values]]
    separator = ["---", *["---:" for _ in k_values]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for method_name in ("standard_rag", "berag"):
        method_rows = rows_by_method.get(method_name, {})
        cells = [
            method_label(method_name),
            *[
                metric_pair_cell(method_rows.get(k_value), percentile_name)
                for k_value in k_values
            ],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_csv(
    *,
    rows_by_method: dict[str, dict[int, dict[str, object]]],
    k_values: list[int],
) -> str:
    lines = [
        "method,k,status,num_rows_used,p50_ttft_ms,p50_tpot_ms,"
        "p90_ttft_ms,p90_tpot_ms"
    ]
    for method_name in ("standard_rag", "berag"):
        method_rows = rows_by_method.get(method_name, {})
        for k_value in k_values:
            record = method_rows.get(k_value, {"status": "missing"})
            lines.append(
                ",".join(
                    [
                        method_label(method_name),
                        str(k_value),
                        str(record.get("status", "missing")),
                        str(record.get("num_rows_used", 0)),
                        str(record.get("p50_ttft_ms", "")),
                        str(record.get("p50_tpot_ms", "")),
                        str(record.get("p90_ttft_ms", "")),
                        str(record.get("p90_tpot_ms", "")),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def summarize_predictions(
    *,
    output_dir: Path,
    k_values: list[int],
    skip_first_rows: int,
) -> dict[str, object]:
    rows_by_method: dict[str, dict[int, dict[str, object]]] = {}
    for method_name in ("standard_rag", "berag"):
        rows_by_method[method_name] = {}
        for k_value in k_values:
            prediction_path = (
                output_dir / method_name / f"k{k_value}" / "predictions.jsonl"
            )
            all_rows = read_jsonl(prediction_path)
            rows = all_rows[skip_first_rows:]
            ttft_values = [
                float(row["ttft_s"]) * 1000
                for row in rows
                if isinstance(row.get("ttft_s"), (int, float))
            ]
            tpot_values = [
                float(row["tpot_s"]) * 1000
                for row in rows
                if isinstance(row.get("tpot_s"), (int, float))
            ]
            if not rows or not ttft_values or not tpot_values:
                rows_by_method[method_name][k_value] = {
                    "status": "missing",
                    "prediction_path": str(prediction_path),
                    "num_rows_total": len(all_rows),
                    "num_rows_used": len(rows),
                    "skipped_first_rows": skip_first_rows,
                }
                continue
            rows_by_method[method_name][k_value] = {
                "status": "ok",
                "prediction_path": str(prediction_path),
                "num_rows_total": len(all_rows),
                "num_rows_used": len(rows),
                "skipped_first_rows": skip_first_rows,
                "p50_ttft_ms": percentile(ttft_values, 50),
                "p50_tpot_ms": percentile(tpot_values, 50),
                "p90_ttft_ms": percentile(ttft_values, 90),
                "p90_tpot_ms": percentile(tpot_values, 90),
            }

    payload: dict[str, object] = {
        "units": "milliseconds",
        "skipped_first_rows_per_run": skip_first_rows,
        "k_values": k_values,
        "methods": rows_by_method,
    }
    write_json(output_dir / "summary_metrics.json", payload)
    write_text(
        output_dir / "summary_p50.md",
        "# P50 TTFT, TPOT\n\n"
        "Values are milliseconds. The first row of each run is ignored.\n\n"
        + build_markdown_table(
            rows_by_method=rows_by_method,
            k_values=k_values,
            percentile_name="p50",
        ),
    )
    write_text(
        output_dir / "summary_p90.md",
        "# P90 TTFT, TPOT\n\n"
        "Values are milliseconds. The first row of each run is ignored.\n\n"
        + build_markdown_table(
            rows_by_method=rows_by_method,
            k_values=k_values,
            percentile_name="p90",
        ),
    )
    write_text(
        output_dir / "summary_metrics.csv",
        build_csv(rows_by_method=rows_by_method, k_values=k_values),
    )
    return payload


def command_record(cmd: list[str]) -> dict[str, object]:
    return {
        "argv": cmd,
        "shell": shlex.join(cmd),
    }


def run_command(label: str, cmd: list[str], summary: dict[str, object],
                summary_path: Path) -> None:
    print(f"[compare] starting {label}")
    print(f"[compare] command: {shlex.join(cmd)}")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    start = time.perf_counter()
    run_record: dict[str, object] = {
        "name": label,
        "command": command_record(cmd),
        "started_at": started_at,
        "status": "running",
    }
    runs = summary.setdefault("runs", [])
    assert isinstance(runs, list)
    runs.append(run_record)
    write_json(summary_path, summary)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        run_record["status"] = "failed"
        run_record["returncode"] = exc.returncode
        run_record["duration_s"] = time.perf_counter() - start
        write_json(summary_path, summary)
        raise
    run_record["status"] = "ok"
    run_record["returncode"] = 0
    run_record["duration_s"] = time.perf_counter() - start
    write_json(summary_path, summary)
    print(f"[compare] finished {label} in {run_record['duration_s']:.2f}s")


def normalize_paths(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(
            Path("my_outputs")
            / "experiments"
            / f"berag_vs_rag_online_bs1_{model_slug(args.model)}_{timestamp}"
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    args.output_dir = str(output_dir)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    args.data_dir = str(data_dir)

    if args.query_image_path:
        query_image_path = Path(args.query_image_path)
        if not query_image_path.is_absolute():
            query_image_path = REPO_ROOT / query_image_path
        args.query_image_path = str(query_image_path)


def validate_args(args: argparse.Namespace) -> None:
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive.")
    if args.summary_skip_first_rows < 0:
        raise ValueError("--summary-skip-first-rows must be non-negative.")
    if not args.no_rag_truncation:
        truncate_tokens = args.rag_truncate_prompt_tokens
        if truncate_tokens is None:
            truncate_tokens = args.max_model_len - args.max_tokens
        if truncate_tokens < 0:
            raise ValueError(
                "RAG truncation token budget is negative; increase "
                "--max-model-len, reduce --max-tokens, or pass "
                "--no-rag-truncation."
            )


def main() -> None:
    args = parse_args()
    normalize_paths(args)
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_k_values(args.k_values)

    commands = {
        "validate": build_validate_cmd(args),
        "standard_rag": build_rag_cmd(args),
        "berag": build_berag_cmd(args),
    }
    config: dict[str, object] = {
        "model": args.model,
        "model_slug": model_slug(args.model),
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "k_values": k_values,
        "max_examples": args.max_examples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "query_image_path": args.query_image_path,
        "num_accumulator_rows": args.num_accumulator_rows,
        "request_batch_size": args.request_batch_size,
        "rag_truncation_side": args.rag_truncation_side,
        "summary_skip_first_rows": args.summary_skip_first_rows,
        "run_order": args.run_order,
        "commands": {
            name: command_record(cmd) for name, cmd in commands.items()
        },
    }
    summary: dict[str, object] = {
        **config,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output_dir / "comparison_config.json", config)
    summary_path = output_dir / "comparison_summary.json"
    write_json(summary_path, summary)

    if args.dry_run:
        summary["status"] = "dry_run"
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(summary_path, summary)
        print(f"[compare] dry run wrote: {output_dir}")
        return

    if not args.skip_validation:
        run_command("validate", commands["validate"], summary, summary_path)

    ordered_names = (
        ["standard_rag", "berag"]
        if args.run_order == "rag-first"
        else ["berag", "standard_rag"]
    )
    try:
        for name in ordered_names:
            run_command(name, commands[name], summary, summary_path)
    except subprocess.CalledProcessError:
        summary["status"] = "failed"
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(summary_path, summary)
        raise

    summary["status"] = "ok"
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["metrics_summary"] = summarize_predictions(
        output_dir=output_dir,
        k_values=k_values,
        skip_first_rows=args.summary_skip_first_rows,
    )
    write_json(summary_path, summary)
    print(f"[compare] done: {output_dir}")
    print(f"[compare] standard RAG: {output_dir / 'standard_rag'}")
    print(f"[compare] BERAG: {output_dir / 'berag'}")
    print(f"[compare] P50 summary: {output_dir / 'summary_p50.md'}")


if __name__ == "__main__":
    main()
