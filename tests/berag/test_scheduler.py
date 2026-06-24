# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from pathlib import Path

import pytest
import torch

from vllm.berag import BeragChildMetadata
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import (
    BeragGroupState,
    BeragRowAllocator,
    Scheduler,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import (
    BeragModelRunnerOutput,
    BeragRowPoolTelemetry,
    ModelRunnerOutput,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test


def make_local_opt_model(tmp_path: Path) -> str:
    model_dir = tmp_path / "tiny-opt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 16,
                "ffn_dim": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "max_position_embeddings": 128,
            }
        )
    )
    return str(model_dir)


def create_local_scheduler(
    tmp_path: Path,
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    block_size: int = 16,
    num_blocks: int = 10000,
) -> Scheduler:
    model_config = ModelConfig(
        model=make_local_opt_model(tmp_path),
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def make_berag_child_request(
    branch_id: int,
    *,
    num_branches: int = 3,
    prompt_len: int = 2,
    max_tokens: int = 4,
    pruning_top_p: float = 0.8,
) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return Request(
        request_id=f"parent:berag:{branch_id}",
        prompt_token_ids=[branch_id] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
        berag_child=BeragChildMetadata(
            group_id="parent",
            parent_request_id="parent",
            branch_id=branch_id,
            num_branches=num_branches,
            parent_prompt_len=1,
            prior_token_index=0,
            pruning_top_p=pruning_top_p,
        ),
    )


def make_berag_model_output(
    *,
    completed_branch_ids: list[int],
    prior_scores: dict[int, float] | None = None,
    sampled_token_id: int | None = None,
    sampled_token_logprobs: dict[int, float] | None = None,
    free_rows: int = 397,
    live_rows: int = 3,
) -> ModelRunnerOutput:
    req_ids = [f"parent:berag:{branch_id}" for branch_id in completed_branch_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[],
        berag_outputs=[
            BeragModelRunnerOutput(
                group_id="parent",
                step_id=0,
                completed_branch_ids=completed_branch_ids,
                prior_scores=prior_scores,
                sampled_token_id=sampled_token_id,
                sampled_token_logprobs=sampled_token_logprobs,
            )
        ],
        berag_row_pool=BeragRowPoolTelemetry(
            total_rows=400,
            free_rows=free_rows,
            live_rows=live_rows,
        ),
    )


def test_berag_row_allocator_reuses_released_rows():
    allocator = BeragRowAllocator(2)

    assert allocator.allocate() == 0
    assert allocator.allocate() == 1
    assert allocator.free_count == 0

    allocator.release([0])

    assert allocator.free_count == 1
    assert allocator.allocate() == 0


def test_berag_group_state_registers_children():
    group = BeragGroupState(
        group_id="parent",
        parent_request_id="parent",
        num_branches=3,
        pruning_top_p=0.8,
    )
    request = make_berag_child_request(branch_id=1)

    group.register_child(request)

    assert group.child_request_ids == {1: "parent:berag:1"}
    assert group.active_branch_ids == {1}
    assert group.prior_token_indices == {1: 0}


def test_berag_scheduler_waits_for_all_children_before_scheduling(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=3, max_num_batched_tokens=6
    )

    scheduler.add_request(make_berag_child_request(0, num_branches=3))

    scheduler_output = scheduler.schedule()

    assert scheduler_output.total_num_scheduled_tokens == 0
    assert scheduler_output.scheduled_berag_shards is None

    scheduler.add_request(make_berag_child_request(1, num_branches=3))
    scheduler.add_request(make_berag_child_request(2, num_branches=3))

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {
        "parent:berag:0": 2,
        "parent:berag:1": 2,
        "parent:berag:2": 2,
    }
    assert scheduler_output.scheduled_berag_shards
    assert set(scheduler_output.scheduled_berag_shards[0].branch_ids) == {0, 1, 2}


def test_berag_top_p_pruning_keeps_at_least_one_branch(tmp_path):
    scheduler = create_local_scheduler(tmp_path)
    group = BeragGroupState(
        group_id="parent",
        parent_request_id="parent",
        num_branches=3,
        pruning_top_p=0.8,
    )
    group.active_branch_ids = {0, 1, 2}
    group.log_posterior = {
        0: math.log(0.6),
        1: math.log(0.3),
        2: math.log(0.1),
    }

    assert scheduler._select_berag_pruned_branches(group) == [2]

    group.pruning_top_p = 0.0
    assert scheduler._select_berag_pruned_branches(group) == [1, 2]


def test_berag_scheduler_defers_final_sampling_until_priors_arrive(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    requests = [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]
    for request in requests:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    shard = scheduler_output.scheduled_berag_shards[0]
    assert shard.group_id == "parent"
    assert shard.branch_ids == [0, 1]
    assert shard.prior_req_ids == ["parent:berag:0", "parent:berag:1"]
    assert shard.is_final_shard
    assert not shard.sample_on_completion

    outputs = scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: -1.0},
        ),
    )

    assert all(not engine_outputs.outputs for engine_outputs in outputs.values())
    assert scheduler.berag_groups["parent"].pending_finalize

    finalize_output = scheduler.schedule()
    finalize_shard = finalize_output.scheduled_berag_shards[0]
    assert finalize_output.total_num_scheduled_tokens == 0
    assert finalize_shard.is_final_shard
    assert finalize_shard.sample_on_completion


def test_berag_final_shard_commits_parent_output_and_releases_rows(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    requests = [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]
    for request in requests:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: -1.0},
        ),
    )
    finalize_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        finalize_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    parent_outputs = outputs[0].outputs
    assert len(parent_outputs) == 1
    assert parent_outputs[0].request_id == "parent"
    assert parent_outputs[0].new_token_ids == [42]
    assert scheduler.berag_row_allocator.free_count == 400
    assert scheduler.berag_release_rows
    assert scheduler.berag_committed_tokens


def test_berag_schedules_decode_after_first_shared_token(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    requests = [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]
    for request in requests:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: -1.0},
        ),
    )
    finalize_output = scheduler.schedule()
    scheduler.update_from_output(
        finalize_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    decode_output = scheduler.schedule()

    assert decode_output.berag_committed_tokens
    assert decode_output.berag_release_rows
    assert decode_output.num_scheduled_tokens == {
        "parent:berag:0": 1,
        "parent:berag:1": 1,
    }
    assert decode_output.scheduled_berag_shards
    assert decode_output.scheduled_berag_shards[0].sample_on_completion
