# SPDX-License-Identifier: Apache-2.0
"""Validate prepared NarrativeQA K-subset JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_scripts.narrativeqa_benchmark_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    parse_k_values,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="my_outputs/data/NarrativeQA")
    parser.add_argument(
        "--k-values",
        default=",".join(str(k_value) for k_value in DEFAULT_K_VALUES),
    )
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def validate_row(row: dict[str, Any], k_value: int, path: Path, index: int) -> None:
    prefix = f"{path}:{index + 1}"
    if row.get("k") != k_value:
        raise ValueError(f"{prefix}: expected k={k_value}, found {row.get('k')}")
    chunk_ids = row.get("chunk_ids")
    chunks = row.get("chunks")
    if not isinstance(chunk_ids, list) or len(chunk_ids) != k_value:
        raise ValueError(f"{prefix}: expected exactly {k_value} chunk_ids")
    if not isinstance(chunks, list) or len(chunks) != k_value:
        raise ValueError(f"{prefix}: expected exactly {k_value} chunks")
    gold_chunk_id = row.get("gold_chunk_id")
    if gold_chunk_id not in chunk_ids:
        raise ValueError(f"{prefix}: gold chunk is missing from chunk_ids")
    gold_position = row.get("gold_position")
    if not isinstance(gold_position, int):
        raise ValueError(f"{prefix}: gold_position must be an int")
    if gold_position < 0 or gold_position >= min(50, k_value):
        raise ValueError(f"{prefix}: gold_position is outside the first 50 chunks")
    if chunk_ids[gold_position] != gold_chunk_id:
        raise ValueError(f"{prefix}: gold_position does not point to gold chunk")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    k_values = parse_k_values(args.k_values)
    rows_by_k: dict[int, list[dict[str, Any]]] = {}

    for k_value in k_values:
        path = data_dir / f"narrativeqa_k{k_value}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared data file: {path}")
        rows = read_jsonl(path, limit=args.max_examples)
        for index, row in enumerate(rows):
            validate_row(row, k_value, path, index)
        rows_by_k[k_value] = rows
        print(f"[validate] k={k_value} rows={len(rows)} ok")

    min_k = min(k_values)
    base_rows = {row["example_id"]: row for row in rows_by_k[min_k]}
    for k_value in k_values:
        if k_value == min_k:
            continue
        for row in rows_by_k[k_value]:
            example_id = row["example_id"]
            if example_id not in base_rows:
                raise ValueError(f"k={k_value}: missing base row for {example_id}")
            base_chunk_ids = base_rows[example_id]["chunk_ids"]
            if row["chunk_ids"][:min_k] != base_chunk_ids:
                raise ValueError(
                    f"k={k_value}: chunk list is not nested for {example_id}"
                )

    print("[validate] nested K subsets ok")


if __name__ == "__main__":
    main()
