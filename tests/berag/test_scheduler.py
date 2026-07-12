# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from pathlib import Path

import pytest
import torch

from vllm.berag import BeragChildMetadata
from vllm.config import (
    BeragConfig,
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
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
    model_dir.mkdir(exist_ok=True)
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
    berag_prior_mode: str = "module",
    berag_group_trace_path: str | None = None,
    berag_group_trace_full_posterior: bool = False,
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
        async_scheduling=False,
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
        berag_config=BeragConfig(
            prior_mode=berag_prior_mode,
            group_trace_path=berag_group_trace_path,
            group_trace_full_posterior=berag_group_trace_full_posterior,
        ),
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
    parent_id: str = "parent",
    num_branches: int = 3,
    prompt_len: int = 2,
    max_tokens: int = 4,
    pruning_top_p: float = 0.8,
    mm_features: list[MultiModalFeatureSpec] | None = None,
) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return Request(
        request_id=f"{parent_id}:berag:{branch_id}",
        prompt_token_ids=[branch_id] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=mm_features,
        block_hasher=get_request_block_hasher(16, sha256),
        berag_child=BeragChildMetadata(
            group_id=parent_id,
            parent_request_id=parent_id,
            branch_id=branch_id,
            num_branches=num_branches,
            parent_prompt_len=1,
            prior_token_index=0,
            pruning_top_p=pruning_top_p,
        ),
    )


def make_berag_model_output(
    *,
    parent_id: str = "parent",
    completed_branch_ids: list[int],
    prior_scores: dict[int, float] | None = None,
    sampled_token_id: int | None = None,
    sampled_token_logprobs: dict[int, float] | None = None,
    step_id: int = 0,
    free_rows: int = 400,
    live_rows: int = 0,
) -> ModelRunnerOutput:
    req_ids = [f"{parent_id}:berag:{branch_id}" for branch_id in completed_branch_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[],
        berag_outputs=[
            BeragModelRunnerOutput(
                group_id=parent_id,
                step_id=step_id,
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
    shard = scheduler_output.scheduled_berag_shards[0]
    assert set(shard.branch_ids) == {0, 1, 2}
    assert shard.direct_mix
    assert shard.evidence_row_ids == []
    assert shard.mix_row_ids == []


def test_berag_complete_group_schedules_shared_encoder_input_once(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=3, max_num_batched_tokens=6
    )
    scheduler.max_num_encoder_input_tokens = 4
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=4)
    shared_image_id = "shared-image"

    for branch_id in range(3):
        scheduler.add_request(
            make_berag_child_request(
                branch_id,
                num_branches=3,
                mm_features=[
                    MultiModalFeatureSpec(
                        data={},
                        modality="image",
                        identifier=shared_image_id,
                        mm_position=PlaceholderRange(offset=0, length=1),
                    )
                ],
            )
        )

    scheduler_output = scheduler.schedule()

    assert scheduler_output.scheduled_encoder_inputs == {
        "parent:berag:0": [0],
    }
    assert scheduler.encoder_cache_manager.cached[shared_image_id] == {
        "parent:berag:0",
        "parent:berag:1",
        "parent:berag:2",
    }
    assert scheduler_output.scheduled_berag_shards
    assert scheduler_output.scheduled_berag_shards[0].direct_mix


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


def test_berag_scheduler_samples_with_worker_priors_in_same_pass(tmp_path):
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
    assert shard.direct_mix
    assert shard.sample_on_completion
    assert shard.log_posterior == []

    outputs = scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: -1.0},
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    assert outputs[0].outputs[0].request_id == "parent"
    assert outputs[0].outputs[0].new_token_ids == [42]
    assert not scheduler.berag_groups["parent"].pending_finalize


def test_uniform_berag_prior_does_not_wait_for_worker_prior_scores(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        berag_prior_mode="uniform",
    )
    for request in [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]:
        scheduler.add_request(request)

    group = scheduler.berag_groups["parent"]
    assert group.prior_scores == {0: 0.0, 1: 0.0}
    assert group.priors_ready

    scheduler_output = scheduler.schedule()
    shard = scheduler_output.scheduled_berag_shards[0]

    assert shard.is_final_shard
    assert shard.direct_mix
    assert shard.sample_on_completion
    assert shard.evidence_row_ids == []
    assert shard.mix_row_ids == []
    assert shard.log_posterior == pytest.approx([-math.log(2), -math.log(2)])

    scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    assert not scheduler.berag_groups["parent"].pending_finalize


def test_berag_row_reservation_uses_adjusted_computed_tokens(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        berag_prior_mode="uniform",
    )
    scheduler.berag_row_allocator = BeragRowAllocator(1)
    scheduler.add_request(
        make_berag_child_request(
            0,
            num_branches=1,
            prompt_len=10,
            max_tokens=1,
            pruning_top_p=1.0,
        )
    )
    request = scheduler.requests["parent:berag:0"]
    assert request.num_computed_tokens == 0

    assert not scheduler._try_reserve_berag_rows(
        request,
        num_computed_tokens=9,
        num_new_tokens=1,
        reserved_mixture_groups=set(),
        reserved_branch_rows=set(),
    )


def test_berag_shard_has_explicit_group_scoped_fields(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        berag_prior_mode="uniform",
    )
    for request in [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    shard = scheduler_output.scheduled_berag_shards[0]

    assert shard.group_id == "parent"
    assert shard.scheduled_req_ids == ["parent:berag:0", "parent:berag:1"]
    assert shard.scheduled_branch_ids == [0, 1]
    assert shard.prior_req_ids == ["parent:berag:0", "parent:berag:1"]
    assert shard.prior_branch_ids == [0, 1]
    assert shard.evidence_branch_ids == [0, 1]
    assert shard.evidence_row_ids == []
    assert shard.mix_req_ids == ["parent:berag:0", "parent:berag:1"]
    assert shard.mix_branch_ids == [0, 1]
    assert shard.mix_row_ids == []
    assert shard.mixture_row_id == -1
    assert shard.direct_mix


def test_berag_scheduler_uses_leftover_capacity_for_partial_group(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=12,
        max_num_batched_tokens=32,
        berag_prior_mode="uniform",
    )
    for parent_index in range(3):
        parent_id = f"parent-{parent_index}"
        for branch_id in range(5):
            scheduler.add_request(
                make_berag_child_request(
                    branch_id,
                    parent_id=parent_id,
                    num_branches=5,
                    prompt_len=1,
                    pruning_top_p=1.0,
                )
            )

    scheduler_output = scheduler.schedule()

    assert len(scheduler_output.scheduled_berag_shards) == 3
    assert set(scheduler_output.num_scheduled_tokens) == {
        f"parent-{parent_index}:berag:{branch_id}"
        for parent_index, branch_count in ((0, 5), (1, 5), (2, 2))
        for branch_id in range(branch_count)
    }
    for shard in scheduler_output.scheduled_berag_shards[:2]:
        assert shard.direct_mix
        assert shard.scheduled_branch_ids == [0, 1, 2, 3, 4]
        assert shard.evidence_row_ids == []
        assert shard.mix_row_ids == []
    partial_shard = scheduler_output.scheduled_berag_shards[2]
    assert partial_shard.group_id == "parent-2"
    assert not partial_shard.direct_mix
    assert partial_shard.scheduled_branch_ids == [0, 1]
    assert partial_shard.evidence_branch_ids == [0, 1]
    assert partial_shard.mixture_row_id >= 0
    assert len(partial_shard.evidence_row_ids) == 2
    assert not partial_shard.is_final_shard
    assert not partial_shard.sample_on_completion


def test_berag_partial_scheduler_uses_available_sequence_budget(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        berag_prior_mode="uniform",
    )
    for branch_id in range(3):
        scheduler.add_request(
            make_berag_child_request(
                branch_id,
                num_branches=3,
                prompt_len=1,
                pruning_top_p=1.0,
            )
        )

    scheduler_output = scheduler.schedule()

    assert len(scheduler_output.scheduled_berag_shards) == 1
    shard = scheduler_output.scheduled_berag_shards[0]
    assert not shard.direct_mix
    assert shard.scheduled_branch_ids == [0, 1]
    assert shard.evidence_branch_ids == [0, 1]
    assert shard.mixture_row_id >= 0
    assert len(shard.evidence_row_ids) == 2
    assert not shard.is_final_shard
    assert not shard.sample_on_completion
    assert "parent:berag:2" not in scheduler_output.num_scheduled_tokens


def test_berag_chunk_without_evidence_is_scheduled_as_partial_work(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=1,
        max_num_batched_tokens=1,
        berag_prior_mode="uniform",
    )
    scheduler.add_request(
        make_berag_child_request(
            0,
            num_branches=1,
            prompt_len=3,
            max_tokens=1,
            pruning_top_p=1.0,
        )
    )

    scheduler_output = scheduler.schedule()

    shard = scheduler_output.scheduled_berag_shards[0]
    assert not shard.direct_mix
    assert shard.scheduled_branch_ids == [0]
    assert shard.evidence_branch_ids == []
    assert shard.evidence_row_ids == []
    assert shard.mixture_row_id == -1
    assert not shard.is_final_shard
    assert not shard.sample_on_completion
    assert scheduler.berag_groups["parent"].completed_branch_ids == set()


def test_berag_scheduler_emits_multiple_direct_final_shards(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=2
    )
    for parent_id in ("parent-a", "parent-b"):
        scheduler.add_request(
            make_berag_child_request(
                0,
                parent_id=parent_id,
                num_branches=1,
                prompt_len=1,
                pruning_top_p=1.0,
            )
        )

    first_output = scheduler.schedule()
    assert [shard.group_id for shard in first_output.scheduled_berag_shards] == [
        "parent-a",
        "parent-b",
    ]
    assert all(shard.direct_mix for shard in first_output.scheduled_berag_shards)
    assert all(
        shard.sample_on_completion
        for shard in first_output.scheduled_berag_shards
    )

    outputs = scheduler.update_from_output(
        first_output,
        ModelRunnerOutput(
            req_ids=["parent-a:berag:0", "parent-b:berag:0"],
            req_id_to_index={"parent-a:berag:0": 0, "parent-b:berag:0": 1},
            sampled_token_ids=[],
            berag_outputs=[
                BeragModelRunnerOutput(
                    group_id="parent-a",
                    step_id=0,
                    completed_branch_ids=[0],
                    prior_scores={0: 0.0},
                    sampled_token_id=11,
                    sampled_token_logprobs={0: -0.1},
                ),
                BeragModelRunnerOutput(
                    group_id="parent-b",
                    step_id=0,
                    completed_branch_ids=[0],
                    prior_scores={0: 0.0},
                    sampled_token_id=12,
                    sampled_token_logprobs={0: -0.1},
                ),
            ],
            berag_row_pool=BeragRowPoolTelemetry(
                total_rows=400,
                free_rows=scheduler.berag_row_allocator.free_count,
                live_rows=400 - scheduler.berag_row_allocator.free_count,
            ),
        ),
    )

    assert [output.request_id for output in outputs[0].outputs] == [
        "parent-a",
        "parent-b",
    ]


def test_berag_scheduler_prefers_in_progress_partial_group(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=4, max_num_batched_tokens=4
    )
    scheduler.max_num_scheduled_tokens = 2
    for branch_id in range(3):
        scheduler.add_request(
            make_berag_child_request(
                branch_id,
                parent_id="parent-a",
                num_branches=3,
                prompt_len=1,
                pruning_top_p=1.0,
            )
        )
    scheduler.add_request(
        make_berag_child_request(
            0,
            parent_id="parent-b",
            num_branches=1,
            prompt_len=1,
            pruning_top_p=1.0,
        )
    )

    first_output = scheduler.schedule()
    assert [shard.group_id for shard in first_output.scheduled_berag_shards] == [
        "parent-a",
    ]
    shard = first_output.scheduled_berag_shards[0]
    assert shard.scheduled_branch_ids == [0, 1]
    assert not shard.direct_mix
    assert "parent-b:berag:0" not in first_output.num_scheduled_tokens


def test_berag_k50_partial_shard_does_not_require_complete_group(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=256,
        max_num_batched_tokens=256,
        berag_prior_mode="uniform",
    )
    for branch_id in range(50):
        scheduler.add_request(
            make_berag_child_request(
                branch_id,
                num_branches=50,
                prompt_len=23,
                pruning_top_p=1.0,
            )
        )

    scheduler_output = scheduler.schedule()

    assert len(scheduler_output.scheduled_berag_shards) == 1
    shard = scheduler_output.scheduled_berag_shards[0]
    scheduled_branch_count = len(shard.scheduled_branch_ids)
    assert 0 < scheduled_branch_count < 50
    assert shard.scheduled_branch_ids == list(range(scheduled_branch_count))
    evidence_branch_count = len(shard.evidence_branch_ids)
    assert 0 < evidence_branch_count <= scheduled_branch_count
    assert shard.evidence_branch_ids == list(range(evidence_branch_count))
    assert not shard.direct_mix
    assert shard.mixture_row_id >= 0
    assert len(shard.evidence_row_ids) == evidence_branch_count
    assert not shard.is_final_shard
    assert not shard.sample_on_completion


def test_berag_scheduler_raises_when_no_partial_shard_can_fit_rows(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        berag_prior_mode="uniform",
    )
    scheduler.berag_row_allocator = BeragRowAllocator(1)
    for branch_id in range(3):
        scheduler.add_request(
            make_berag_child_request(
                branch_id,
                num_branches=3,
                prompt_len=1,
                pruning_top_p=1.0,
            )
        )

    with pytest.raises(RuntimeError, match="No BERAG shard can be scheduled"):
        scheduler.schedule()


def test_berag_group_trace_writes_compact_posterior(tmp_path):
    trace_path = tmp_path / "group_trace.jsonl"
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        berag_prior_mode="uniform",
        berag_group_trace_path=str(trace_path),
    )
    for request in [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [row["event"] for row in rows]

    assert "schedule_shard" in events
    assert "receive_evidence" in events
    assert "posterior_update" in events
    assert "commit_token" in events

    posterior_row = next(row for row in rows if row["event"] == "posterior_update")
    assert posterior_row["posterior_top_branch_id"] == 0
    assert posterior_row["posterior_top5"]
    assert posterior_row["posterior_count"] == 2
    assert "posterior_full" not in posterior_row


def test_berag_group_trace_full_posterior_is_optional(tmp_path):
    trace_path = tmp_path / "group_trace.jsonl"
    scheduler = create_local_scheduler(
        tmp_path,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        berag_prior_mode="uniform",
        berag_group_trace_path=str(trace_path),
        berag_group_trace_full_posterior=True,
    )
    for request in [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    posterior_row = next(row for row in rows if row["event"] == "posterior_update")

    assert set(posterior_row["posterior_full"]) == {"0", "1"}
    assert sum(posterior_row["posterior_full"].values()) == pytest.approx(1.0)
    assert posterior_row["sampled_token_logprobs"] == {"0": -0.1, "1": -2.0}


def test_berag_direct_shard_commits_parent_output_without_rows(tmp_path):
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
    outputs = scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: -1.0},
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    parent_outputs = outputs[0].outputs
    assert len(parent_outputs) == 1
    assert parent_outputs[0].request_id == "parent"
    assert parent_outputs[0].new_token_ids == [42]
    assert scheduler.berag_row_allocator.free_count == 400
    assert scheduler.berag_committed_tokens


def test_berag_final_output_includes_prior_posterior_info(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    for request in [
        make_berag_child_request(
            0, num_branches=2, max_tokens=1, pruning_top_p=1.0
        ),
        make_berag_child_request(
            1, num_branches=2, max_tokens=1, pruning_top_p=1.0
        ),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: -1.0},
            sampled_token_id=42,
            sampled_token_logprobs={0: -2.0, 1: -0.1},
        ),
    )

    parent_output = outputs[0].outputs[0]
    info = parent_output.berag_info
    assert parent_output.finish_reason is not None
    assert info is not None
    expected_prior = Scheduler._normalize_logs({0: 0.0, 1: -1.0})
    expected_posterior = Scheduler._normalize_logs(
        {0: expected_prior[0] - 2.0, 1: expected_prior[1] - 0.1}
    )
    assert info["num_branches"] == 2
    assert info["log_prior_by_branch"] == pytest.approx(
        [expected_prior[0], expected_prior[1]]
    )
    assert info["log_posterior_by_branch"] == pytest.approx(
        [expected_posterior[0], expected_posterior[1]]
    )
    assert info["prior_max_branch_id"] == 0
    assert info["posterior_max_branch_id"] == 1
    assert info["prior_sorted_branch_ids"] == [0, 1]
    assert info["posterior_sorted_branch_ids"] == [1, 0]
    assert info["active_branch_ids"] == [0, 1]
    assert info["pruned_branch_ids"] == []


def test_berag_final_output_marks_pruned_posterior_as_none(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    for request in [
        make_berag_child_request(
            0, num_branches=2, max_tokens=2, pruning_top_p=0.8
        ),
        make_berag_child_request(
            1, num_branches=2, max_tokens=2, pruning_top_p=0.8
        ),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: 0.0},
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -10.0},
        ),
    )
    assert scheduler.berag_groups["parent"].active_branch_ids == {0}

    decode_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        decode_output,
        make_berag_model_output(
            completed_branch_ids=[0],
            sampled_token_id=43,
            sampled_token_logprobs={0: -0.2},
            step_id=1,
        ),
    )

    info = outputs[0].outputs[0].berag_info
    assert info is not None
    assert info["log_prior_by_branch"] == pytest.approx([-math.log(2), -math.log(2)])
    assert info["log_posterior_by_branch"] == [0.0, None]
    assert info["posterior_max_branch_id"] == 0
    assert info["posterior_sorted_branch_ids"] == [0]
    assert info["active_branch_ids"] == [0]
    assert info["pruned_branch_ids"] == [1]


def test_berag_final_shard_updates_posterior_and_prunes_branch(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    requests = [
        make_berag_child_request(0, num_branches=2, pruning_top_p=0.8),
        make_berag_child_request(1, num_branches=2, pruning_top_p=0.8),
    ]
    for request in requests:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output,
        make_berag_model_output(
            completed_branch_ids=[0, 1],
            prior_scores={0: 0.0, 1: 0.0},
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -10.0},
        ),
    )

    assert outputs[0].outputs[0].request_id == "parent"
    group = scheduler.berag_groups["parent"]
    assert group.step_id == 1
    assert group.active_branch_ids == {0}
    assert set(group.log_posterior) == {0}
    assert group.log_posterior[0] == pytest.approx(0.0)

    decode_output = scheduler.schedule()

    assert decode_output.num_scheduled_tokens == {"parent:berag:0": 1}
    assert decode_output.berag_committed_tokens[0].req_ids == [
        "parent:berag:0",
        "parent:berag:1",
    ]


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
            sampled_token_id=42,
            sampled_token_logprobs={0: -0.1, 1: -2.0},
        ),
    )

    decode_output = scheduler.schedule()

    assert decode_output.berag_committed_tokens
    assert decode_output.num_scheduled_tokens == {
        "parent:berag:0": 1,
        "parent:berag:1": 1,
    }
    assert decode_output.scheduled_berag_shards
    assert decode_output.scheduled_berag_shards[0].sample_on_completion


def test_berag_scheduler_rejects_mixing_ordinary_and_berag_requests(tmp_path):
    scheduler = create_local_scheduler(tmp_path)

    def make_ordinary_request() -> Request:
        init_none_hash(sha256)
        return Request(
            request_id="ordinary",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(16, sha256),
        )

    ordinary = make_ordinary_request()
    scheduler.add_request(ordinary)
    with pytest.raises(ValueError, match="mixed with ordinary"):
        scheduler.add_request(make_berag_child_request(0, num_branches=1))

    scheduler = create_local_scheduler(tmp_path)
    scheduler.add_request(make_berag_child_request(0, num_branches=1))
    with pytest.raises(ValueError, match="Ordinary requests"):
        scheduler.add_request(make_ordinary_request())


def test_berag_scheduler_rejects_worker_row_telemetry_mismatch(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    for request in [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    with pytest.raises(RuntimeError, match="row telemetry mismatch"):
        scheduler.update_from_output(
            scheduler_output,
            make_berag_model_output(
                completed_branch_ids=[0, 1],
                prior_scores={0: 0.0, 1: 0.0},
                free_rows=399,
                live_rows=1,
            ),
        )


def test_berag_scheduler_ignores_stale_step_outputs(tmp_path):
    scheduler = create_local_scheduler(
        tmp_path, max_num_seqs=2, max_num_batched_tokens=4
    )
    for request in [
        make_berag_child_request(0, num_branches=2, pruning_top_p=1.0),
        make_berag_child_request(1, num_branches=2, pruning_top_p=1.0),
    ]:
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    stale_output = make_berag_model_output(
        completed_branch_ids=[0, 1],
        prior_scores={0: 0.0, 1: 0.0},
    )
    stale_output.berag_outputs[0].step_id = 1
    stale_output.berag_row_pool = BeragRowPoolTelemetry(
        total_rows=400,
        free_rows=scheduler.berag_row_allocator.free_count,
        live_rows=400 - scheduler.berag_row_allocator.free_count,
    )

    outputs = scheduler.update_from_output(scheduler_output, stale_output)

    assert all(not engine_outputs.outputs for engine_outputs in outputs.values())
    group = scheduler.berag_groups["parent"]
    assert group.prior_scores == {}
    assert group.completed_branch_ids == set()
