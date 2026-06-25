# SPDX-License-Identifier: Apache-2.0
"""Prepare static NarrativeQA K-subsets for RAG and BERAG benchmarks."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_scripts.narrativeqa_benchmark_utils import (  # noqa: E402
    ANSWER_FIELDS,
    DEFAULT_K_VALUES,
    DEFAULT_SYSTEM_PROMPT,
    DOCUMENT_ID_FIELDS,
    DOCUMENT_TEXT_FIELDS,
    GOLD_CHUNK_FIELDS,
    QUERY_ID_FIELDS,
    QUERY_TEXT_FIELDS,
    base_doc_id,
    build_nested_chunk_ids,
    chunk_sort_key,
    first_text_field,
    length_summary,
    make_longbench_prompt,
    make_standard_rag_context,
    parse_k_values,
    render_qwen_chat_prompt,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="illuin-conteb/narrative-qa")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--query-split", default="test")
    parser.add_argument("--document-split", default="all")
    parser.add_argument("--output-dir", default="my_outputs/data/NarrativeQA")
    parser.add_argument(
        "--k-values",
        default=",".join(str(k_value) for k_value in DEFAULT_K_VALUES),
    )
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gold-window", type=int, default=50)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-token-lengths", action="store_true")
    return parser.parse_args()


def load_dataset_config(dataset_name: str, config: str, revision: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required. Run: source my_scripts/activate_env.sh && "
            "uv pip install datasets"
        ) from exc

    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(dataset_name, config, **kwargs)


def iter_dataset_splits(dataset: Any) -> Iterable[tuple[str, Any]]:
    if hasattr(dataset, "items"):
        yield from dataset.items()
    else:
        yield "default", dataset


def dataset_fingerprints(dataset: Any) -> dict[str, str | None]:
    return {
        split: getattr(split_dataset, "_fingerprint", None)
        for split, split_dataset in iter_dataset_splits(dataset)
    }


def select_split(dataset: Any, preferred_split: str) -> tuple[str, Any]:
    splits = dict(iter_dataset_splits(dataset))
    if preferred_split in splits:
        return preferred_split, splits[preferred_split]
    for fallback in ("test", "validation", "dev", "train"):
        if fallback in splits:
            return fallback, splits[fallback]
    split_name = next(iter(splits))
    return split_name, splits[split_name]


def selected_document_splits(
    dataset: Any,
    document_split: str,
) -> Iterable[tuple[str, Any]]:
    if document_split == "all":
        yield from iter_dataset_splits(dataset)
    else:
        yield select_split(dataset, document_split)


def collect_documents(
    dataset: Any,
    *,
    document_split: str,
) -> tuple[dict[str, str], dict[str, int], int]:
    documents: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    skipped_rows = 0

    for split_name, split_dataset in selected_document_splits(dataset, document_split):
        count = 0
        for row in split_dataset:
            chunk_id = first_text_field(row, DOCUMENT_ID_FIELDS)
            chunk_text = first_text_field(row, DOCUMENT_TEXT_FIELDS)
            if not chunk_id or not chunk_text:
                skipped_rows += 1
                continue
            documents.setdefault(chunk_id, chunk_text)
            count += 1
        split_counts[split_name] = count
    return documents, split_counts, skipped_rows


def build_same_document_map(documents: Mapping[str, str]) -> dict[str, list[str]]:
    same_doc_map: dict[str, list[str]] = defaultdict(list)
    for chunk_id in documents:
        same_doc_map[base_doc_id(chunk_id)].append(chunk_id)
    for chunk_ids in same_doc_map.values():
        chunk_ids.sort(key=chunk_sort_key)
    return same_doc_map


def make_examples(
    *,
    query_dataset: Any,
    query_split_name: str,
    documents: Mapping[str, str],
    k_values: list[int],
    max_examples: int,
    seed: int,
    gold_window: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_chunk_ids = sorted(documents, key=chunk_sort_key)
    same_doc_map = build_same_document_map(documents)
    examples: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    seen_example_ids: set[str] = set()

    for row_index, row in enumerate(query_dataset):
        if len(examples) >= max_examples:
            break

        question = first_text_field(row, QUERY_TEXT_FIELDS)
        answer = first_text_field(row, ANSWER_FIELDS)
        gold_chunk_id = first_text_field(row, GOLD_CHUNK_FIELDS)
        if not question:
            skipped["missing_question"] += 1
            continue
        if not answer:
            skipped["missing_answer"] += 1
            continue
        if not gold_chunk_id:
            skipped["missing_gold_chunk"] += 1
            continue
        if gold_chunk_id not in documents:
            skipped["gold_chunk_not_found"] += 1
            continue

        example_id = first_text_field(row, QUERY_ID_FIELDS)
        if not example_id:
            example_id = f"{query_split_name}_{row_index}"
        if example_id in seen_example_ids:
            example_id = f"{example_id}_{row_index}"
        seen_example_ids.add(example_id)

        same_document_chunk_ids = same_doc_map[base_doc_id(gold_chunk_id)]
        nested = build_nested_chunk_ids(
            example_id=example_id,
            gold_chunk_id=gold_chunk_id,
            same_document_chunk_ids=same_document_chunk_ids,
            all_chunk_ids=all_chunk_ids,
            k_values=k_values,
            seed=seed,
            gold_window=gold_window,
        )
        if nested is None:
            skipped["insufficient_chunks"] += 1
            continue

        chunk_ids, gold_position = nested
        examples.append(
            {
                "example_id": example_id,
                "question": question,
                "answer": answer,
                "gold_chunk_id": gold_chunk_id,
                "gold_position": gold_position,
                "chunk_ids": chunk_ids,
            }
        )

    return examples, dict(skipped)


def load_tokenizer(args: argparse.Namespace) -> Any | None:
    if args.skip_token_lengths or not args.tokenizer:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for token length summaries."
        ) from exc
    return AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )


def summarize_prompt_lengths(
    *,
    examples: list[dict[str, Any]],
    documents: Mapping[str, str],
    k_values: list[int],
    tokenizer: Any | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    token_lengths: dict[str, list[int]] = {str(k_value): [] for k_value in k_values}
    word_lengths: dict[str, list[int]] = {str(k_value): [] for k_value in k_values}
    total = len(examples) * len(k_values)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=total,
            desc="[data] summarizing prompt lengths",
            unit="prompt",
        )
    else:
        print(f"[data] summarizing prompt lengths for {total} prompts")

    try:
        for example in examples:
            for k_value in k_values:
                chunk_ids = example["chunk_ids"][:k_value]
                chunks = [documents[chunk_id] for chunk_id in chunk_ids]
                context = make_standard_rag_context(chunks)
                user_prompt = make_longbench_prompt(context, example["question"])
                word_lengths[str(k_value)].append(len(user_prompt.split()))
                if tokenizer is not None:
                    rendered = render_qwen_chat_prompt(
                        tokenizer,
                        user_prompt,
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                    )
                    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
                    token_lengths[str(k_value)].append(len(token_ids))
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    word_summary = {
        k_value: length_summary(lengths) for k_value, lengths in word_lengths.items()
    }
    if tokenizer is None:
        return None, word_summary
    token_summary = {
        k_value: length_summary(lengths) for k_value, lengths in token_lengths.items()
    }
    return token_summary, word_summary


def iter_output_rows(
    *,
    examples: list[dict[str, Any]],
    documents: Mapping[str, str],
    k_value: int,
) -> Iterable[dict[str, Any]]:
    for example in examples:
        chunk_ids = example["chunk_ids"][:k_value]
        yield {
            "example_id": example["example_id"],
            "question": example["question"],
            "answer": example["answer"],
            "gold_chunk_id": example["gold_chunk_id"],
            "gold_position": example["gold_position"],
            "k": k_value,
            "chunk_ids": chunk_ids,
            "chunks": [documents[chunk_id] for chunk_id in chunk_ids],
        }


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading {args.dataset_name}/documents")
    documents_dataset = load_dataset_config(
        args.dataset_name,
        "documents",
        args.revision,
    )
    print(f"[data] loading {args.dataset_name}/queries")
    queries_dataset = load_dataset_config(args.dataset_name, "queries", args.revision)

    documents, document_split_counts, skipped_document_rows = collect_documents(
        documents_dataset,
        document_split=args.document_split,
    )
    query_split_name, query_dataset = select_split(queries_dataset, args.query_split)
    examples, skipped_query_rows = make_examples(
        query_dataset=query_dataset,
        query_split_name=query_split_name,
        documents=documents,
        k_values=k_values,
        max_examples=args.max_examples,
        seed=args.seed,
        gold_window=args.gold_window,
    )

    print(
        f"[data] built {len(examples)} examples from query split "
        f"{query_split_name!r}; documents={len(documents)}"
    )

    row_counts: dict[str, int] = {}
    for k_value in k_values:
        output_path = output_dir / f"narrativeqa_k{k_value}.jsonl"
        rows = list(
            iter_output_rows(
                examples=examples,
                documents=documents,
                k_value=k_value,
            )
        )
        write_jsonl(output_path, rows)
        row_counts[str(k_value)] = len(rows)
        print(f"[data] wrote {output_path} rows={len(rows)}")

    tokenizer = load_tokenizer(args)
    token_summary, word_summary = summarize_prompt_lengths(
        examples=examples,
        documents=documents,
        k_values=k_values,
        tokenizer=tokenizer,
    )

    manifest = {
        "seed": args.seed,
        "k_values": k_values,
        "max_examples": args.max_examples,
        "gold_window": args.gold_window,
        "source": {
            "dataset_name": args.dataset_name,
            "revision": args.revision,
            "documents_config": "documents",
            "queries_config": "queries",
            "document_split": args.document_split,
            "query_split": query_split_name,
            "documents_fingerprints": dataset_fingerprints(documents_dataset),
            "queries_fingerprints": dataset_fingerprints(queries_dataset),
        },
        "row_counts": row_counts,
        "skipped_row_counts": {
            "documents": {"missing_fields": skipped_document_rows},
            "queries": skipped_query_rows,
        },
        "document_split_counts": document_split_counts,
        "token_length_summaries": token_summary,
        "word_length_summaries": word_summary,
        "gold_position_summary": length_summary(
            [example["gold_position"] for example in examples]
        ),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[data] wrote {manifest_path}")


if __name__ == "__main__":
    main()
