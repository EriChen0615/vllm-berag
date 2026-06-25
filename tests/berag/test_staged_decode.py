# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from tests.berag.test_scheduler import (
    create_local_scheduler,
    make_berag_child_request,
)
from vllm import SamplingParams
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.outputs import (
    BeragModelRunnerOutput,
    BeragRowPoolTelemetry,
    ModelRunnerOutput,
)
from vllm.v1.engine import EngineCoreEventType, EngineCoreRequest

pytestmark = pytest.mark.cpu_test


def _branch_req_id(branch_id: int) -> str:
    return f"parent:berag:{branch_id}"


def _row_pool_for_scheduler(scheduler) -> BeragRowPoolTelemetry:
    total_rows = scheduler.berag_config.num_accumulator_rows
    free_rows = scheduler.berag_row_allocator.free_count
    return BeragRowPoolTelemetry(
        total_rows=total_rows,
        free_rows=free_rows,
        live_rows=total_rows - free_rows,
    )


def _berag_model_output(
    scheduler,
    *,
    step_id: int,
    completed_branch_ids: list[int],
    prior_scores: dict[int, float] | None = None,
    sampled_token_id: int | None = None,
) -> ModelRunnerOutput:
    req_ids = [_branch_req_id(branch_id) for branch_id in completed_branch_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[],
        berag_outputs=[
            BeragModelRunnerOutput(
                group_id="parent",
                step_id=step_id,
                completed_branch_ids=completed_branch_ids,
                prior_scores=prior_scores,
                sampled_token_id=sampled_token_id,
                sampled_token_logprobs={
                    branch_id: -0.1 for branch_id in completed_branch_ids
                }
                if sampled_token_id is not None
                else None,
            )
        ],
        berag_row_pool=_row_pool_for_scheduler(scheduler),
    )


def test_berag_staged_decode_samples_after_all_branch_shards(tmp_path):
    scheduler = create_local_scheduler(tmp_path, max_num_seqs=8)
    scheduler.max_num_scheduled_tokens = 2

    for branch_id in range(5):
        scheduler.add_request(
            make_berag_child_request(
                branch_id,
                num_branches=5,
                prompt_len=1,
                max_tokens=2,
                pruning_top_p=1.0,
            )
        )

    first_prefill = scheduler.schedule()
    first_shard = first_prefill.scheduled_berag_shards[0]
    assert first_shard.step_id == 0
    assert first_shard.branch_ids == [0, 1]
    assert not first_shard.is_final_shard
    assert not first_shard.sample_on_completion

    scheduler.update_from_output(
        first_prefill,
        _berag_model_output(
            scheduler,
            step_id=0,
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: 0.0},
        ),
    )

    second_prefill = scheduler.schedule()
    second_shard = second_prefill.scheduled_berag_shards[0]
    assert second_shard.step_id == 0
    assert second_shard.branch_ids == [2, 3]
    assert not second_shard.is_final_shard
    assert not second_shard.sample_on_completion

    scheduler.update_from_output(
        second_prefill,
        _berag_model_output(
            scheduler,
            step_id=0,
            completed_branch_ids=[2, 3],
            prior_scores={2: 0.0, 3: 0.0},
        ),
    )

    last_prefill = scheduler.schedule()
    last_prefill_shard = last_prefill.scheduled_berag_shards[0]
    assert last_prefill_shard.step_id == 0
    assert last_prefill_shard.branch_ids == [4]
    assert last_prefill_shard.is_final_shard
    assert not last_prefill_shard.sample_on_completion

    scheduler.update_from_output(
        last_prefill,
        _berag_model_output(
            scheduler,
            step_id=0,
            completed_branch_ids=[4],
            prior_scores={4: 0.0},
        ),
    )

    finalize_prefill = scheduler.schedule()
    finalize_shard = finalize_prefill.scheduled_berag_shards[0]
    assert finalize_prefill.total_num_scheduled_tokens == 0
    assert finalize_shard.step_id == 0
    assert finalize_shard.branch_ids == [0, 1, 2, 3, 4]
    assert finalize_shard.is_final_shard
    assert finalize_shard.sample_on_completion

    parent_outputs = scheduler.update_from_output(
        finalize_prefill,
        _berag_model_output(
            scheduler,
            step_id=0,
            completed_branch_ids=[0, 1, 2, 3, 4],
            sampled_token_id=11,
        ),
    )
    assert parent_outputs[0].outputs[0].request_id == "parent"
    first_parent_output = parent_outputs[0].outputs[0]
    assert first_parent_output.new_token_ids == [11]
    assert first_parent_output.events is not None
    assert [event.type for event in first_parent_output.events] == [
        EngineCoreEventType.QUEUED,
        EngineCoreEventType.SCHEDULED,
    ]
    group = scheduler.berag_groups["parent"]
    assert group.parent_events_emitted
    assert first_parent_output.events[0].timestamp == group.parent_queued_ts
    assert first_parent_output.events[1].timestamp == group.first_scheduled_ts

    parent_request = EngineCoreRequest(
        request_id="parent",
        external_req_id="parent",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=2, detokenize=False),
        pooling_params=None,
        arrival_time=time.time() - 2.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)
    output_processor.add_request(parent_request, prompt=None)
    engine_core_timestamp = first_parent_output.events[1].timestamp + 1.0
    processed = output_processor.process_outputs(
        [first_parent_output],
        engine_core_timestamp=engine_core_timestamp,
        iteration_stats=IterationStats(),
    )
    parent_request_output = processed.request_outputs[0]
    assert parent_request_output.metrics is not None
    parent_metrics = parent_request_output.metrics
    assert parent_metrics.queued_ts == first_parent_output.events[0].timestamp
    assert parent_metrics.scheduled_ts == first_parent_output.events[1].timestamp
    assert parent_metrics.first_token_ts == engine_core_timestamp
    assert parent_metrics.first_token_ts > parent_metrics.scheduled_ts

    first_decode = scheduler.schedule()
    assert first_decode.berag_committed_tokens
    assert first_decode.berag_release_rows
    first_decode_shard = first_decode.scheduled_berag_shards[0]
    assert first_decode_shard.step_id == 1
    assert first_decode_shard.branch_ids == [0, 1]
    assert not first_decode_shard.is_final_shard

    scheduler.update_from_output(
        first_decode,
        _berag_model_output(
            scheduler,
            step_id=1,
            completed_branch_ids=[0, 1],
        ),
    )

    second_decode = scheduler.schedule()
    second_decode_shard = second_decode.scheduled_berag_shards[0]
    assert second_decode_shard.step_id == 1
    assert second_decode_shard.branch_ids == [2, 3]
    assert not second_decode_shard.is_final_shard

    scheduler.update_from_output(
        second_decode,
        _berag_model_output(
            scheduler,
            step_id=1,
            completed_branch_ids=[2, 3],
        ),
    )

    final_decode = scheduler.schedule()
    final_decode_shard = final_decode.scheduled_berag_shards[0]
    assert final_decode_shard.step_id == 1
    assert final_decode_shard.branch_ids == [0, 1, 2, 3, 4]
    assert final_decode_shard.is_final_shard
    assert final_decode_shard.sample_on_completion

    final_outputs = scheduler.update_from_output(
        final_decode,
        _berag_model_output(
            scheduler,
            step_id=1,
            completed_branch_ids=[0, 1, 2, 3, 4],
            sampled_token_id=22,
        ),
    )
    assert final_outputs[0].outputs[0].request_id == "parent"
    final_parent_output = final_outputs[0].outputs[0]
    assert final_parent_output.new_token_ids == [22]
    assert final_parent_output.events is None
    assert "parent" not in scheduler.berag_groups
