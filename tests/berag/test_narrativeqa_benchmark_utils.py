# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types

import pytest

from my_scripts.narrativeqa_benchmark_utils import (
    aggregate_prediction_metrics,
    base_doc_id,
    build_nested_chunk_ids,
    build_prediction_row,
    make_longbench_prompt,
    make_narrativeqa_prompt,
    make_standard_rag_context,
    read_jsonl,
    render_qwen_chat_prompt,
    write_jsonl,
)

pytestmark = pytest.mark.cpu_test


def test_base_doc_id_strips_only_numeric_chunk_suffix():
    assert base_doc_id("book_12") == "book"
    assert base_doc_id("book_part_a") == "book_part_a"


def test_nested_chunk_ids_keep_gold_in_first_50_and_preserve_prefixes():
    same_doc = [f"story_0_{index}" for index in range(12)]
    gold_chunk_id = "story_0_7"
    distractors = [
        f"story_{doc_index}_{chunk_index}"
        for doc_index in range(1, 80)
        for chunk_index in range(3)
    ]
    all_chunk_ids = same_doc + distractors

    nested = build_nested_chunk_ids(
        example_id="example-1",
        gold_chunk_id=gold_chunk_id,
        same_document_chunk_ids=same_doc,
        all_chunk_ids=all_chunk_ids,
        k_values=[50, 75, 100, 150, 200],
        seed=123,
    )

    assert nested is not None
    chunk_ids, gold_position = nested
    assert len(chunk_ids) == 200
    assert 0 <= gold_position < 50
    assert chunk_ids[gold_position] == gold_chunk_id
    assert gold_chunk_id in chunk_ids[:50]
    assert chunk_ids[:75][:50] == chunk_ids[:50]
    assert chunk_ids[:100][:75] == chunk_ids[:75]
    for same_doc_chunk_id in same_doc:
        assert same_doc_chunk_id in chunk_ids


class FakeTokenizer:
    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        assert add_generation_prompt
        self.last_messages = messages
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        if isinstance(user_content, list):
            user_content = "".join(
                "<image>" if item["type"] == "image" else item["text"]
                for item in user_content
            )
        return (
            "<|im_start|>system\n"
            f"{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def test_prompt_rendering_uses_qwen_chat_template_and_longbench_prompt():
    context = make_standard_rag_context(["alpha", "beta"])
    user_prompt = make_longbench_prompt(context, "Who wins?")
    rendered = render_qwen_chat_prompt(FakeTokenizer(), user_prompt)

    assert "<|im_start|>system" in rendered
    assert "You are a helpful assistant" in rendered
    assert "[Chunk 1]\nalpha" in rendered
    assert "[Chunk 2]\nbeta" in rendered
    assert "Question: Who wins?" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_image_prompt_rendering_adds_image_block_and_response_format():
    tokenizer = FakeTokenizer()
    context = make_standard_rag_context(["alpha"])
    user_prompt = make_narrativeqa_prompt(
        context,
        "Who wins?",
        include_image=True,
    )

    rendered = render_qwen_chat_prompt(
        tokenizer,
        user_prompt,
        include_image=True,
    )

    assert tokenizer.last_messages[1]["content"][0] == {"type": "image"}
    assert tokenizer.last_messages[1]["content"][1]["type"] == "text"
    assert "First, describe the image" in rendered
    assert "Image:" in rendered
    assert "Answer:" in rendered


class FakeCompletion:
    text = "short answer"
    token_ids = [11, 12, 13]
    finish_reason = "length"


class FakeMetrics:
    first_token_latency = 1.25
    queued_ts = 10.0
    scheduled_ts = 10.5
    first_token_ts = 12.0
    last_token_ts = 13.0


class FakeOutput:
    outputs = [FakeCompletion()]
    prompt_token_ids = [1, 2, 3, 4]
    metrics = FakeMetrics()
    finished = True


def test_prediction_rows_and_aggregate_metrics(monkeypatch):
    example = {
        "example_id": "ex",
        "k": 50,
        "question": "Question?",
        "answer": "Reference",
    }
    row = build_prediction_row(example=example, output=FakeOutput())

    assert row["prompt_tokens"] == 4
    assert row["output_tokens"] == 3
    assert row["ttft_s"] == pytest.approx(1.25)
    assert row["queued_s"] == pytest.approx(0.5)
    assert row["prefill_s"] == pytest.approx(1.5)
    assert row["decode_s"] == pytest.approx(1.0)
    assert row["tpot_s"] == pytest.approx(0.5)

    fake_sacrebleu = types.SimpleNamespace(
        corpus_bleu=lambda predictions, references: types.SimpleNamespace(score=7.5)
    )
    monkeypatch.setitem(__import__("sys").modules, "sacrebleu", fake_sacrebleu)
    metrics = aggregate_prediction_metrics([row], wall_time_s=2.0)
    assert metrics["num_requests"] == 1
    assert metrics["requests_per_second"] == pytest.approx(0.5)
    assert metrics["corpus_bleu"] == pytest.approx(7.5)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": "two"}])
    assert read_jsonl(path) == [{"a": 1}, {"b": "two"}]
    assert read_jsonl(path, limit=1) == [{"a": 1}]
