# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    BeragCommittedTokens,
    BeragReleaseRows,
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    ScheduledBeragShard,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


@dataclass
class BeragRowAllocator:
    num_rows: int
    free_rows: deque[int] = field(init=False)

    def __post_init__(self) -> None:
        self.free_rows = deque(range(self.num_rows))

    @property
    def free_count(self) -> int:
        return len(self.free_rows)

    def allocate(self) -> int:
        return self.free_rows.popleft()

    def release(self, rows: Iterable[int]) -> None:
        for row in rows:
            self.free_rows.append(row)


@dataclass
class BeragGroupState:
    group_id: str
    parent_request_id: str
    num_branches: int
    pruning_top_p: float
    debug: bool = False
    child_request_ids: dict[int, str] = field(default_factory=dict)
    active_branch_ids: set[int] = field(default_factory=set)
    prior_token_indices: dict[int, int] = field(default_factory=dict)
    prior_scores: dict[int, float] = field(default_factory=dict)
    log_posterior: dict[int, float] = field(default_factory=dict)
    step_id: int = 0
    completed_branch_ids: set[int] = field(default_factory=set)
    mixture_row_id: int | None = None
    branch_row_ids: dict[int, int] = field(default_factory=dict)
    pending_finalize: bool = False
    step_started: bool = False
    parent_queued_ts: float | None = None
    first_scheduled_ts: float | None = None
    parent_events_emitted: bool = False

    def register_child(self, request: Request) -> None:
        meta = request.berag_child
        assert meta is not None
        self.child_request_ids[meta.branch_id] = request.request_id
        self.active_branch_ids.add(meta.branch_id)
        self.prior_token_indices[meta.branch_id] = meta.prior_token_index
        self.debug = self.debug or meta.debug
        for event in request.events:
            if event.type == EngineCoreEventType.QUEUED:
                if (
                    self.parent_queued_ts is None
                    or event.timestamp < self.parent_queued_ts
                ):
                    self.parent_queued_ts = event.timestamp
                break

    @property
    def all_children_registered(self) -> bool:
        return len(self.child_request_ids) == self.num_branches

    @property
    def priors_ready(self) -> bool:
        return self.active_branch_ids.issubset(self.prior_scores)

    @property
    def step_evidence_ready(self) -> bool:
        return self.active_branch_ids.issubset(self.completed_branch_ids)

    def reset_step(self) -> None:
        self.step_id += 1
        self.completed_branch_ids.clear()
        self.mixture_row_id = None
        self.branch_row_ids.clear()
        self.pending_finalize = False
        self.step_started = False


@dataclass
class BeragBranchSchedulePlan:
    request: Request
    branch_id: int
    num_computed_tokens: int
    num_new_tokens: int
    new_computed_blocks: KVCacheBlocks | None = None
    num_new_local_computed_tokens: int = 0
    was_waiting: bool = False


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        self.berag_config = vllm_config.berag_config
        self.berag_groups: dict[str, BeragGroupState] = {}
        self.berag_group_order: deque[str] = deque()
        self.berag_row_allocator = BeragRowAllocator(
            self.berag_config.num_accumulator_rows
        )
        self.berag_release_rows: list[BeragReleaseRows] = []
        self.berag_committed_tokens: list[BeragCommittedTokens] = []
        self.berag_group_trace_path = (
            Path(self.berag_config.group_trace_path)
            if self.berag_config.group_trace_path
            else None
        )
        if self.berag_group_trace_path is not None:
            self.berag_group_trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self.berag_config.enabled:
            if (
                self.parallel_config.tensor_parallel_size != 1
                or self.parallel_config.pipeline_parallel_size != 1
                or self.parallel_config.data_parallel_size != 1
            ):
                raise ValueError("BERAG scheduler supports only single-GPU execution.")
            if self.scheduler_config.async_scheduling:
                raise ValueError("BERAG scheduler does not support async scheduling.")
            if self.vllm_config.speculative_config is not None:
                raise ValueError("BERAG scheduler does not support speculative decode.")

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_uncached_common_prefix_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass

            # Marconi cache admission optimization:
            # cache common prefixes by scheduling num_new_tokens = common prefix length
            if (
                num_uncached_common_prefix_tokens >= block_size
                and num_new_tokens > num_uncached_common_prefix_tokens
            ):
                num_new_tokens = num_uncached_common_prefix_tokens
                # keep alignment to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
        return num_new_tokens

    def _make_empty_scheduler_output(
        self,
        *,
        scheduled_berag_shards: list[ScheduledBeragShard] | None = None,
    ) -> SchedulerOutput:
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = self.finished_req_ids
        output.berag_release_rows = self._take_berag_release_rows()
        output.berag_committed_tokens = self._take_berag_committed_tokens()
        output.scheduled_berag_shards = scheduled_berag_shards
        return output

    def _take_berag_release_rows(self) -> list[BeragReleaseRows] | None:
        if not self.berag_release_rows:
            return None
        rows = self.berag_release_rows
        self.berag_release_rows = []
        return rows

    def _take_berag_committed_tokens(self) -> list[BeragCommittedTokens] | None:
        if not self.berag_committed_tokens:
            return None
        commands = self.berag_committed_tokens
        self.berag_committed_tokens = []
        return commands

    @staticmethod
    def _berag_debug(group: BeragGroupState, message: str, *args: Any) -> None:
        if not group.debug:
            return
        logger.info(
            "[BERAG debug] scheduler group=%s step=%d " + message,
            group.group_id,
            group.step_id,
            *args,
        )

    def _berag_posterior_trace(
        self, group: BeragGroupState
    ) -> dict[str, Any]:
        if not group.log_posterior:
            return {
                "posterior_count": 0,
                "posterior_entropy": None,
                "posterior_max": None,
                "posterior_top_branch_id": None,
                "posterior_top5": [],
            }
        probs = {
            branch_id: math.exp(log_prob)
            for branch_id, log_prob in group.log_posterior.items()
        }
        ordered = sorted(probs, key=lambda branch_id: (-probs[branch_id], branch_id))
        entropy = -sum(
            prob * group.log_posterior[branch_id]
            for branch_id, prob in probs.items()
        )
        trace: dict[str, Any] = {
            "posterior_count": len(probs),
            "posterior_entropy": entropy,
            "posterior_max": probs[ordered[0]],
            "posterior_top_branch_id": ordered[0],
            "posterior_top5": [
                [branch_id, probs[branch_id]] for branch_id in ordered[:5]
            ],
        }
        if self.berag_config.group_trace_full_posterior:
            trace["posterior_full"] = {
                str(branch_id): probs[branch_id]
                for branch_id in sorted(probs)
            }
        return trace

    def _write_berag_group_trace(
        self, group: BeragGroupState, event: str, **fields: Any
    ) -> None:
        if self.berag_group_trace_path is None:
            return
        row: dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            "group_id": group.group_id,
            "parent_request_id": group.parent_request_id,
            "step_id": group.step_id,
            "num_branches": group.num_branches,
            "active_branch_ids": sorted(group.active_branch_ids),
            "active_branch_count": len(group.active_branch_ids),
            "completed_branch_ids": sorted(group.completed_branch_ids),
            "completed_branch_count": len(group.completed_branch_ids),
            "priors_ready": group.priors_ready,
            "step_evidence_ready": group.step_evidence_ready,
            "pending_finalize": group.pending_finalize,
            "row_pool_free": self.berag_row_allocator.free_count,
            "row_pool_live": (
                self.berag_config.num_accumulator_rows
                - self.berag_row_allocator.free_count
            ),
        }
        row.update(fields)
        row.update(self._berag_posterior_trace(group))
        with self.berag_group_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _collect_ready_berag_finalizes(self) -> list[ScheduledBeragShard]:
        shards: list[ScheduledBeragShard] = []
        for group_id in list(self.berag_group_order):
            group = self.berag_groups.get(group_id)
            if group is None:
                continue
            if (
                group.pending_finalize
                and group.all_children_registered
                and group.priors_ready
                and group.step_evidence_ready
                and group.mixture_row_id is not None
            ):
                self._ensure_berag_posterior(group)
                branch_ids = sorted(group.active_branch_ids)
                req_ids = [
                    group.child_request_ids[branch_id] for branch_id in branch_ids
                ]
                row_ids = [
                    group.branch_row_ids[branch_id] for branch_id in branch_ids
                ]
                self._berag_debug(
                    group,
                    "emit deferred final shard branches=%s rows=%s "
                    "log_posterior=%s",
                    branch_ids,
                    row_ids,
                    [group.log_posterior[branch_id] for branch_id in branch_ids],
                )
                self._write_berag_group_trace(
                    group,
                    "schedule_deferred_final_shard",
                    mix_req_ids=req_ids,
                    mix_branch_ids=branch_ids,
                    mix_row_ids=row_ids,
                    mixture_row_id=group.mixture_row_id,
                    is_final_shard=True,
                    sample_on_completion=True,
                )
                shards.append(ScheduledBeragShard(
                    group_id=group.group_id,
                    step_id=group.step_id,
                    mixture_row_id=group.mixture_row_id,
                    scheduled_req_ids=[],
                    scheduled_branch_ids=[],
                    prior_req_ids=[],
                    prior_branch_ids=[],
                    prior_token_indices=[],
                    evidence_branch_ids=[],
                    evidence_row_ids=[],
                    mix_req_ids=req_ids,
                    mix_branch_ids=branch_ids,
                    mix_row_ids=row_ids,
                    log_posterior=[
                        group.log_posterior[branch_id] for branch_id in branch_ids
                    ],
                    is_final_shard=True,
                    sample_on_completion=True,
                    debug=group.debug,
                ))
        return shards

    @staticmethod
    def _berag_range_covers_prior_from(
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> bool:
        meta = request.berag_child
        if meta is None:
            return False
        start = num_computed_tokens
        end = start + num_new_tokens
        return start <= meta.prior_token_index < end

    @classmethod
    def _berag_range_covers_prior(
        cls, request: Request, num_new_tokens: int
    ) -> bool:
        return cls._berag_range_covers_prior_from(
            request,
            request.num_computed_tokens,
            num_new_tokens,
        )

    @staticmethod
    def _berag_emits_evidence_from(
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> bool:
        if request.berag_child is None:
            return False
        return num_computed_tokens + num_new_tokens >= request.num_tokens

    @classmethod
    def _berag_emits_evidence(cls, request: Request, num_new_tokens: int) -> bool:
        return cls._berag_emits_evidence_from(
            request,
            request.num_computed_tokens,
            num_new_tokens,
        )

    def _berag_group_is_ready_to_schedule(self, request: Request) -> bool:
        meta = request.berag_child
        if meta is None:
            return True
        group = self.berag_groups[meta.group_id]
        if group.all_children_registered:
            return True
        self._berag_debug(
            group,
            "defer scheduling req=%s branch=%d until all children are "
            "registered children=%d/%d",
            request.request_id,
            meta.branch_id,
            len(group.child_request_ids),
            group.num_branches,
        )
        return False

    def _berag_rows_needed(self, request: Request) -> int:
        meta = request.berag_child
        assert meta is not None
        group = self.berag_groups[meta.group_id]
        needed = 0 if group.mixture_row_id is not None else 1
        if meta.branch_id not in group.branch_row_ids:
            needed += 1
        return needed

    def _try_reserve_berag_rows(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        reserved_mixture_groups: set[str],
        reserved_branch_rows: set[tuple[str, int]],
    ) -> bool:
        meta = request.berag_child
        if meta is None or not self._berag_emits_evidence_from(
            request,
            num_computed_tokens,
            num_new_tokens,
        ):
            return True
        group = self.berag_groups[meta.group_id]
        needed = 0
        if (
            group.mixture_row_id is None
            and meta.group_id not in reserved_mixture_groups
        ):
            needed += 1
        branch_key = (meta.group_id, meta.branch_id)
        if meta.branch_id not in group.branch_row_ids and (
            branch_key not in reserved_branch_rows
        ):
            needed += 1
        already_reserved = len(reserved_mixture_groups) + len(reserved_branch_rows)
        if self.berag_row_allocator.free_count - already_reserved < needed:
            self._berag_debug(
                group,
                "row backpressure branch=%d needed=%d free=%d "
                "already_reserved=%d",
                meta.branch_id,
                needed,
                self.berag_row_allocator.free_count,
                already_reserved,
            )
            return False
        if group.mixture_row_id is None:
            reserved_mixture_groups.add(meta.group_id)
        if meta.branch_id not in group.branch_row_ids:
            reserved_branch_rows.add(branch_key)
        return True

    def _build_scheduled_berag_shards(
        self,
        num_scheduled_tokens: dict[str, int],
    ) -> list[ScheduledBeragShard] | None:
        shards: list[ScheduledBeragShard] = []
        group_to_scheduled_req_ids: dict[str, list[str]] = defaultdict(list)
        group_to_evidence_req_ids: dict[str, list[str]] = defaultdict(list)
        group_to_prior_req_ids: dict[str, list[str]] = defaultdict(list)
        for req_id, num_tokens in num_scheduled_tokens.items():
            request = self.requests[req_id]
            if request.berag_child is None:
                continue
            meta = request.berag_child
            group_to_scheduled_req_ids[meta.group_id].append(req_id)
            if self._berag_emits_evidence(request, num_tokens):
                group_to_evidence_req_ids[meta.group_id].append(req_id)
            if self._berag_range_covers_prior(request, num_tokens):
                group_to_prior_req_ids[meta.group_id].append(req_id)

        ordered_group_ids = [
            group_id
            for group_id in self.berag_group_order
            if (
                group_id in group_to_scheduled_req_ids
                or group_id in group_to_evidence_req_ids
                or group_id in group_to_prior_req_ids
            )
        ]
        for group_id in ordered_group_ids:
            group = self.berag_groups[group_id]
            if not group.all_children_registered:
                raise RuntimeError(
                    "BERAG scheduled a shard before all child requests were "
                    f"registered: group={group_id}, "
                    f"children={len(group.child_request_ids)}/"
                    f"{group.num_branches}."
                )
            scheduled_req_ids = group_to_scheduled_req_ids.get(group_id, [])
            evidence_req_ids = group_to_evidence_req_ids.get(group_id, [])
            if scheduled_req_ids:
                group.step_started = True
            if group.first_scheduled_ts is None and scheduled_req_ids:
                for req_id in scheduled_req_ids:
                    for event in reversed(self.requests[req_id].events):
                        if event.type == EngineCoreEventType.SCHEDULED:
                            group.first_scheduled_ts = event.timestamp
                            break
                    if group.first_scheduled_ts is not None:
                        break
            scheduled_branch_ids: list[int] = []
            for req_id in scheduled_req_ids:
                meta = self.requests[req_id].berag_child
                assert meta is not None
                scheduled_branch_ids.append(meta.branch_id)

            evidence_branch_ids: list[int] = []
            evidence_row_ids: list[int] = []
            rows_needed = 0
            if evidence_req_ids and group.mixture_row_id is None:
                rows_needed += 1
            for req_id in evidence_req_ids:
                meta = self.requests[req_id].berag_child
                assert meta is not None
                if meta.branch_id not in group.branch_row_ids:
                    rows_needed += 1
            if self.berag_row_allocator.free_count < rows_needed:
                raise RuntimeError(
                    "BERAG row reservation invariant violated: "
                    f"group={group_id}, evidence_req_ids={evidence_req_ids}, "
                    f"rows_needed={rows_needed}, "
                    f"free_rows={self.berag_row_allocator.free_count}."
                )
            if evidence_req_ids and group.mixture_row_id is None:
                group.mixture_row_id = self.berag_row_allocator.allocate()
            for req_id in evidence_req_ids:
                meta = self.requests[req_id].berag_child
                assert meta is not None
                evidence_branch_ids.append(meta.branch_id)
                if meta.branch_id not in group.branch_row_ids:
                    group.branch_row_ids[meta.branch_id] = (
                        self.berag_row_allocator.allocate()
                    )
                evidence_row_ids.append(group.branch_row_ids[meta.branch_id])

            completed_after = group.completed_branch_ids | set(evidence_branch_ids)
            is_final = group.active_branch_ids.issubset(completed_after)
            priors_ready = group.priors_ready
            sample_on_completion = bool(is_final and priors_ready)
            if is_final and not sample_on_completion:
                group.pending_finalize = True

            mix_req_ids: list[str] = []
            mix_branch_ids: list[int] = []
            mix_row_ids: list[int] = []
            log_posterior: list[float] = []
            if sample_on_completion:
                self._ensure_berag_posterior(group)
                mix_branch_ids = sorted(group.active_branch_ids)
                mix_req_ids = [
                    group.child_request_ids[branch_id]
                    for branch_id in mix_branch_ids
                ]
                mix_row_ids = [
                    group.branch_row_ids[branch_id] for branch_id in mix_branch_ids
                ]
                log_posterior = [
                    group.log_posterior[branch_id] for branch_id in mix_branch_ids
                ]

            prior_req_ids = group_to_prior_req_ids.get(group_id)
            prior_branch_ids: list[int] = []
            prior_token_indices: list[int] = []
            if prior_req_ids:
                for req_id in prior_req_ids:
                    meta = self.requests[req_id].berag_child
                    assert meta is not None
                    prior_branch_ids.append(meta.branch_id)
                    prior_token_indices.append(meta.prior_token_index)
            shard = ScheduledBeragShard(
                group_id=group_id,
                step_id=group.step_id,
                mixture_row_id=group.mixture_row_id
                if group.mixture_row_id is not None
                else -1,
                scheduled_req_ids=scheduled_req_ids,
                scheduled_branch_ids=scheduled_branch_ids,
                prior_req_ids=prior_req_ids or [],
                prior_branch_ids=prior_branch_ids,
                prior_token_indices=prior_token_indices,
                evidence_branch_ids=evidence_branch_ids,
                evidence_row_ids=evidence_row_ids,
                mix_req_ids=mix_req_ids,
                mix_branch_ids=mix_branch_ids,
                mix_row_ids=mix_row_ids,
                log_posterior=log_posterior,
                is_final_shard=is_final,
                sample_on_completion=sample_on_completion,
                debug=group.debug,
            )
            self._berag_debug(
                group,
                "emit shard scheduled_branches=%s evidence_branches=%s "
                "mix_branches=%s reqs=%s evidence_rows=%s mix_rows=%s "
                "mixture_row=%s prior_reqs=%s final=%s sample=%s "
                "pending_finalize=%s completed_before=%s active=%s",
                scheduled_branch_ids,
                evidence_branch_ids,
                mix_branch_ids,
                scheduled_req_ids,
                evidence_row_ids,
                mix_row_ids,
                shard.mixture_row_id,
                prior_req_ids,
                is_final,
                sample_on_completion,
                group.pending_finalize,
                sorted(group.completed_branch_ids),
                sorted(group.active_branch_ids),
            )
            self._write_berag_group_trace(
                group,
                "schedule_shard",
                scheduled_req_ids=scheduled_req_ids,
                scheduled_branch_ids=scheduled_branch_ids,
                evidence_branch_ids=evidence_branch_ids,
                evidence_row_ids=evidence_row_ids,
                mix_req_ids=mix_req_ids,
                mix_branch_ids=mix_branch_ids,
                mix_row_ids=mix_row_ids,
                mixture_row_id=shard.mixture_row_id,
                prior_req_ids=prior_req_ids or [],
                prior_branch_ids=prior_branch_ids,
                prior_token_indices=prior_token_indices or [],
                is_final_shard=is_final,
                sample_on_completion=sample_on_completion,
            )
            shards.append(shard)
        return shards or None

    @staticmethod
    def _normalize_logs(log_values: dict[int, float]) -> dict[int, float]:
        max_log = max(log_values.values())
        denom = max_log + math.log(
            sum(math.exp(value - max_log) for value in log_values.values())
        )
        return {key: value - denom for key, value in log_values.items()}

    def _ensure_berag_posterior(self, group: BeragGroupState) -> None:
        if group.log_posterior:
            return
        group.log_posterior = self._normalize_logs(
            {
                branch_id: group.prior_scores[branch_id]
                for branch_id in group.active_branch_ids
            }
        )

    def _select_berag_pruned_branches(self, group: BeragGroupState) -> list[int]:
        if group.pruning_top_p >= 1.0 or len(group.active_branch_ids) <= 1:
            return []
        probs = {
            branch_id: math.exp(group.log_posterior[branch_id])
            for branch_id in group.active_branch_ids
        }
        ordered = sorted(probs, key=lambda branch_id: (-probs[branch_id], branch_id))
        kept: set[int] = set()
        mass = 0.0
        for branch_id in ordered:
            kept.add(branch_id)
            mass += probs[branch_id]
            if mass >= group.pruning_top_p:
                break
        if not kept:
            kept.add(ordered[0])
        return sorted(group.active_branch_ids - kept)

    def _release_berag_step_rows(self, group: BeragGroupState) -> None:
        row_ids = list(group.branch_row_ids.values())
        if group.mixture_row_id is not None:
            row_ids.append(group.mixture_row_id)
        if not row_ids:
            return
        self.berag_row_allocator.release(row_ids)
        self._berag_debug(group, "release rows=%s", row_ids)
        self.berag_release_rows.append(
            BeragReleaseRows(
                group_id=group.group_id,
                step_id=group.step_id,
                row_ids=row_ids,
                debug=group.debug,
            )
        )

    def _take_berag_parent_events(
        self, group: BeragGroupState
    ) -> list[EngineCoreEvent] | None:
        if group.parent_events_emitted:
            return None
        if group.parent_queued_ts is None or group.first_scheduled_ts is None:
            raise RuntimeError(
                "BERAG parent timing was not captured before parent output: "
                f"group={group.group_id}, queued_ts={group.parent_queued_ts}, "
                f"scheduled_ts={group.first_scheduled_ts}."
            )
        group.parent_events_emitted = True
        return [
            EngineCoreEvent.new_event(
                EngineCoreEventType.QUEUED, group.parent_queued_ts
            ),
            EngineCoreEvent.new_event(
                EngineCoreEventType.SCHEDULED, group.first_scheduled_ts
            ),
        ]

    def _update_berag_from_output(
        self,
        model_runner_output: ModelRunnerOutput,
        outputs: dict[int, list[EngineCoreOutput]],
    ) -> set[Request]:
        stopped_running: set[Request] = set()
        for berag_output in model_runner_output.berag_outputs or []:
            group = self.berag_groups.get(berag_output.group_id)
            if group is None or berag_output.step_id != group.step_id:
                continue
            if berag_output.prior_scores:
                group.prior_scores.update(berag_output.prior_scores)
                self._berag_debug(
                    group,
                    "received prior_scores=%s priors_ready=%s active=%s",
                    berag_output.prior_scores,
                    group.priors_ready,
                    sorted(group.active_branch_ids),
                )
            group.completed_branch_ids.update(berag_output.completed_branch_ids)
            self._berag_debug(
                group,
                "received branch evidence completed_now=%s completed_step=%s "
                "evidence_ready=%s sampled_token=%s",
                berag_output.completed_branch_ids,
                sorted(group.completed_branch_ids),
                group.step_evidence_ready,
                berag_output.sampled_token_id,
            )
            self._write_berag_group_trace(
                group,
                "receive_evidence",
                completed_now=berag_output.completed_branch_ids,
                prior_branch_ids=(
                    sorted(berag_output.prior_scores)
                    if berag_output.prior_scores
                    else []
                ),
                sampled_token_id=berag_output.sampled_token_id,
            )

            if berag_output.sampled_token_id is None:
                continue

            self._ensure_berag_posterior(group)
            if berag_output.sampled_token_logprobs:
                updated = {
                    branch_id: group.log_posterior[branch_id]
                    + berag_output.sampled_token_logprobs[branch_id]
                    for branch_id in group.active_branch_ids
                }
                group.log_posterior = self._normalize_logs(updated)
            self._berag_debug(
                group,
                "posterior updated sampled_token=%s sampled_logprobs=%s "
                "log_posterior=%s",
                berag_output.sampled_token_id,
                berag_output.sampled_token_logprobs,
                group.log_posterior,
            )
            posterior_fields: dict[str, Any] = {}
            if (
                self.berag_config.group_trace_full_posterior
                and berag_output.sampled_token_logprobs
            ):
                posterior_fields["sampled_token_logprobs"] = {
                    str(branch_id): logprob
                    for branch_id, logprob in sorted(
                        berag_output.sampled_token_logprobs.items()
                    )
                }
            self._write_berag_group_trace(
                group,
                "posterior_update",
                sampled_token_id=berag_output.sampled_token_id,
                **posterior_fields,
            )

            sampled_token_id = berag_output.sampled_token_id
            stopped = False
            finish_reason = None
            representative: Request | None = None
            for branch_id in sorted(group.active_branch_ids):
                req_id = group.child_request_ids[branch_id]
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                representative = representative or request
                _, branch_stopped = self._update_request_with_output(
                    request, [sampled_token_id]
                )
                stopped |= branch_stopped
                if branch_stopped:
                    finish_reason = request.get_finished_reason()
                    stopped_running.add(request)

            pruned_branch_ids: list[int] = []
            committed_req_ids = [
                group.child_request_ids[branch_id]
                for branch_id in sorted(group.active_branch_ids)
            ]
            self.berag_committed_tokens.append(
                BeragCommittedTokens(
                    group_id=group.group_id,
                    step_id=group.step_id,
                    req_ids=committed_req_ids,
                    token_id=sampled_token_id,
                    debug=group.debug,
                )
            )
            self._berag_debug(
                group,
                "commit token=%d to reqs=%s stopped=%s finish_reason=%s",
                sampled_token_id,
                committed_req_ids,
                stopped,
                finish_reason,
            )
            if not stopped:
                pruned_branch_ids = self._select_berag_pruned_branches(group)
                for branch_id in pruned_branch_ids:
                    req_id = group.child_request_ids[branch_id]
                    request = self.requests.get(req_id)
                    if request is None or request.is_finished():
                        continue
                    request.status = RequestStatus.FINISHED_IGNORED
                    self._free_request(request)
                    stopped_running.add(request)
                    group.active_branch_ids.remove(branch_id)
                    group.log_posterior.pop(branch_id, None)
                if pruned_branch_ids and group.log_posterior:
                    group.log_posterior = self._normalize_logs(group.log_posterior)
                self._berag_debug(
                    group,
                    "pruned=%s remaining=%s log_posterior=%s",
                    pruned_branch_ids,
                    sorted(group.active_branch_ids),
                    group.log_posterior,
                )

            self._write_berag_group_trace(
                group,
                "commit_token",
                sampled_token_id=sampled_token_id,
                committed_req_ids=committed_req_ids,
                committed_req_count=len(committed_req_ids),
                pruned_branch_ids=pruned_branch_ids,
                stopped=stopped,
                finish_reason=str(finish_reason) if finish_reason else None,
            )

            if stopped:
                self._berag_debug(group, "group stopping; freeing active branches")
                self._release_berag_step_rows(group)
                for branch_id in sorted(group.active_branch_ids):
                    req_id = group.child_request_ids[branch_id]
                    request = self.requests.get(req_id)
                    if request is None:
                        continue
                    if not request.is_finished():
                        request.status = RequestStatus.FINISHED_STOPPED
                    self._free_request(request)
                    stopped_running.add(request)
                self.berag_groups.pop(group.group_id, None)
                try:
                    self.berag_group_order.remove(group.group_id)
                except ValueError:
                    pass
            else:
                self._release_berag_step_rows(group)
                group.reset_step()
                self._berag_debug(
                    group,
                    "advance to next step active=%s",
                    sorted(group.active_branch_ids),
                )

            if representative is not None:
                outputs[representative.client_index].append(
                    EngineCoreOutput(
                        request_id=group.parent_request_id,
                        new_token_ids=[sampled_token_id],
                        finish_reason=finish_reason if stopped else None,
                        new_logprobs=berag_output.logprobs,
                        stop_reason=representative.stop_reason,
                        events=self._take_berag_parent_events(group),
                        trace_headers=representative.trace_headers,
                    )
                )
        return stopped_running

    def _check_berag_row_telemetry(
        self, model_runner_output: ModelRunnerOutput
    ) -> None:
        telemetry = model_runner_output.berag_row_pool
        if telemetry is None:
            return
        total_rows = self.berag_config.num_accumulator_rows
        expected_free = self.berag_row_allocator.free_count
        expected_live = total_rows - expected_free
        if (
            telemetry.total_rows != total_rows
            or telemetry.free_rows != expected_free
            or telemetry.live_rows != expected_live
        ):
            raise RuntimeError(
                "BERAG accumulator row telemetry mismatch: "
                f"worker=(total={telemetry.total_rows}, "
                f"free={telemetry.free_rows}, live={telemetry.live_rows}), "
                f"scheduler=(total={total_rows}, free={expected_free}, "
                f"live={expected_live})."
            )

    def _berag_group_has_pending_work(self, group: BeragGroupState) -> bool:
        if not group.all_children_registered or group.pending_finalize:
            return False
        for branch_id in sorted(group.active_branch_ids - group.completed_branch_ids):
            req_id = group.child_request_ids[branch_id]
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            if (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            ) > 0:
                return True
            if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                return True
        return False

    def _berag_ordered_groups_for_work(self) -> list[BeragGroupState]:
        groups: list[BeragGroupState] = []
        for in_progress in (True, False):
            for group_id in list(self.berag_group_order):
                group = self.berag_groups.get(group_id)
                if group is None or not self._berag_group_has_pending_work(group):
                    continue
                group_in_progress = group.step_started or bool(
                    group.completed_branch_ids
                )
                if group_in_progress == in_progress:
                    groups.append(group)
        return groups

    def _remove_berag_waiting_request(self, request: Request) -> None:
        for queue in (self.waiting, self.skipped_waiting):
            if any(queued is request for queued in queue):
                queue.remove_request(request)
                return

    def _plan_berag_branch_request(
        self,
        request: Request,
        token_budget: int,
        virtual_running_reqs: int,
    ) -> BeragBranchSchedulePlan | None:
        meta = request.berag_child
        assert meta is not None
        if request.status == RequestStatus.RUNNING:
            if self.current_step < request.next_decode_eligible_step:
                return None
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            if num_new_tokens <= 0:
                return None
            return BeragBranchSchedulePlan(
                request=request,
                branch_id=meta.branch_id,
                num_computed_tokens=request.num_computed_tokens,
                num_new_tokens=num_new_tokens,
            )

        if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            return None
        if virtual_running_reqs >= self.max_num_running_reqs:
            return None

        if request.num_computed_tokens == 0:
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            num_computed_tokens = num_new_local_computed_tokens
        else:
            new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        num_new_tokens = request.num_tokens - num_computed_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        if (
            not self.scheduler_config.enable_chunked_prefill
            and num_new_tokens > token_budget
        ):
            return None
        num_new_tokens = min(num_new_tokens, token_budget)
        if num_new_tokens <= 0:
            return None

        return BeragBranchSchedulePlan(
            request=request,
            branch_id=meta.branch_id,
            num_computed_tokens=num_computed_tokens,
            num_new_tokens=num_new_tokens,
            new_computed_blocks=new_computed_blocks,
            num_new_local_computed_tokens=num_new_local_computed_tokens,
            was_waiting=True,
        )

    def _commit_berag_branch_plan(
        self,
        plan: BeragBranchSchedulePlan,
        scheduled_timestamp: float,
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
        scheduled_encoder_inputs: dict[str, list[int]],
        encoder_compute_budget: int,
    ) -> tuple[bool, int]:
        request = plan.request
        num_new_tokens = plan.num_new_tokens
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            (
                encoder_inputs_to_schedule,
                num_new_tokens,
                new_encoder_compute_budget,
                external_load_encoder_input,
            ) = self._try_schedule_encoder_inputs(
                request,
                plan.num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens=1 if self.use_eagle else 0,
            )
            if num_new_tokens != plan.num_new_tokens:
                return False, encoder_compute_budget

        if request.status == RequestStatus.RUNNING:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                return False, encoder_compute_budget
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
            return True, encoder_compute_budget

        if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            return False, encoder_compute_budget
        has_scheduled_reqs = bool(
            self.running
            or scheduled_new_reqs
            or scheduled_resumed_reqs
            or scheduled_running_reqs
        )
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=plan.num_new_local_computed_tokens,
            new_computed_blocks=plan.new_computed_blocks,
            num_lookahead_tokens=self.num_lookahead_tokens,
            has_scheduled_reqs=has_scheduled_reqs,
        )
        if new_blocks is None:
            if request.has_encoder_inputs:
                self.encoder_cache_manager.free(request)
            return False, encoder_compute_budget

        if request.num_computed_tokens == 0:
            if self.kv_cache_manager.log_stats:
                assert self.kv_cache_manager.prefix_cache_stats is not None
                self.kv_cache_manager.prefix_cache_stats.record(
                    num_tokens=request.num_tokens,
                    num_hits=plan.num_new_local_computed_tokens,
                    preempted=request.num_preemptions > 0,
                )
            if request.prefill_stats is not None:
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=plan.num_new_local_computed_tokens,
                    num_external_cached_tokens=0,
                )

        self._remove_berag_waiting_request(request)
        self.running.append(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
        if request.status == RequestStatus.WAITING:
            scheduled_new_reqs.append(request)
        else:
            scheduled_resumed_reqs.append(request)

        req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(
            request.request_id
        )
        num_scheduled_tokens[request.request_id] = num_new_tokens
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = plan.num_computed_tokens
        if plan.num_computed_tokens + num_new_tokens < request.num_tokens:
            self._inflight_prefills.add(request)
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
            encoder_compute_budget = new_encoder_compute_budget
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
        return True, encoder_compute_budget

    def _berag_evidence_branch_ids_for_plans(
        self, plans: list[BeragBranchSchedulePlan]
    ) -> list[int]:
        return [
            plan.branch_id
            for plan in plans
            if self._berag_emits_evidence_from(
                plan.request,
                plan.num_computed_tokens,
                plan.num_new_tokens,
            )
        ]

    def _berag_rows_needed_for_plans(
        self, group: BeragGroupState, plans: list[BeragBranchSchedulePlan]
    ) -> int:
        evidence_branch_ids = self._berag_evidence_branch_ids_for_plans(plans)
        if not evidence_branch_ids:
            return 0
        needed = 0 if group.mixture_row_id is not None else 1
        for branch_id in evidence_branch_ids:
            if branch_id not in group.branch_row_ids:
                needed += 1
        return needed

    def _berag_direct_mix_candidate(
        self, group: BeragGroupState, plans: list[BeragBranchSchedulePlan]
    ) -> bool:
        if not plans:
            return False
        if group.completed_branch_ids or group.branch_row_ids:
            return False
        if group.mixture_row_id is not None:
            return False
        evidence_branch_ids = set(self._berag_evidence_branch_ids_for_plans(plans))
        if not group.active_branch_ids.issubset(evidence_branch_ids):
            return False
        prior_branch_ids = {
            plan.branch_id
            for plan in plans
            if self._berag_range_covers_prior_from(
                plan.request,
                plan.num_computed_tokens,
                plan.num_new_tokens,
            )
        }
        return group.priors_ready or group.active_branch_ids.issubset(
            prior_branch_ids
        )

    def _make_berag_shard_from_committed_plans(
        self, group: BeragGroupState, plans: list[BeragBranchSchedulePlan]
    ) -> ScheduledBeragShard | None:
        scheduled_req_ids = [plan.request.request_id for plan in plans]
        scheduled_branch_ids = [plan.branch_id for plan in plans]
        if scheduled_req_ids:
            group.step_started = True
        if group.first_scheduled_ts is None and scheduled_req_ids:
            for plan in plans:
                for event in reversed(plan.request.events):
                    if event.type == EngineCoreEventType.SCHEDULED:
                        group.first_scheduled_ts = event.timestamp
                        break
                if group.first_scheduled_ts is not None:
                    break

        prior_req_ids: list[str] = []
        prior_branch_ids: list[int] = []
        prior_token_indices: list[int] = []
        for plan in plans:
            if self._berag_range_covers_prior_from(
                plan.request, plan.num_computed_tokens, plan.num_new_tokens
            ):
                meta = plan.request.berag_child
                assert meta is not None
                prior_req_ids.append(plan.request.request_id)
                prior_branch_ids.append(plan.branch_id)
                prior_token_indices.append(meta.prior_token_index)

        evidence_plans = [
            plan
            for plan in plans
            if self._berag_emits_evidence_from(
                plan.request,
                plan.num_computed_tokens,
                plan.num_new_tokens,
            )
        ]
        evidence_branch_ids = [plan.branch_id for plan in evidence_plans]
        completed_after = group.completed_branch_ids | set(evidence_branch_ids)
        is_final = group.active_branch_ids.issubset(completed_after)
        direct_mix_priors_ready = group.priors_ready or (
            group.active_branch_ids.issubset(set(prior_branch_ids))
        )
        direct_mix = bool(
            is_final
            and direct_mix_priors_ready
            and not group.completed_branch_ids
            and not group.branch_row_ids
            and group.mixture_row_id is None
            and group.active_branch_ids.issubset(evidence_branch_ids)
        )
        sample_on_completion = bool(
            is_final and (group.priors_ready or direct_mix)
        )

        evidence_row_ids: list[int] = []
        if evidence_plans and not direct_mix:
            rows_needed = self._berag_rows_needed_for_plans(group, plans)
            if self.berag_row_allocator.free_count < rows_needed:
                return None
            if group.mixture_row_id is None:
                group.mixture_row_id = self.berag_row_allocator.allocate()
            for plan in evidence_plans:
                if plan.branch_id not in group.branch_row_ids:
                    group.branch_row_ids[plan.branch_id] = (
                        self.berag_row_allocator.allocate()
                    )
                evidence_row_ids.append(group.branch_row_ids[plan.branch_id])

        if is_final and not sample_on_completion:
            group.pending_finalize = True

        mix_req_ids: list[str] = []
        mix_branch_ids: list[int] = []
        mix_row_ids: list[int] = []
        log_posterior: list[float] = []
        if sample_on_completion:
            if group.priors_ready:
                self._ensure_berag_posterior(group)
            mix_branch_ids = sorted(group.active_branch_ids)
            mix_req_ids = [
                group.child_request_ids[branch_id] for branch_id in mix_branch_ids
            ]
            if not direct_mix:
                mix_row_ids = [
                    group.branch_row_ids[branch_id] for branch_id in mix_branch_ids
                ]
            if group.log_posterior:
                log_posterior = [
                    group.log_posterior[branch_id] for branch_id in mix_branch_ids
                ]

        shard = ScheduledBeragShard(
            group_id=group.group_id,
            step_id=group.step_id,
            mixture_row_id=-1
            if direct_mix or group.mixture_row_id is None
            else group.mixture_row_id,
            scheduled_req_ids=scheduled_req_ids,
            scheduled_branch_ids=scheduled_branch_ids,
            prior_req_ids=prior_req_ids,
            prior_branch_ids=prior_branch_ids,
            prior_token_indices=prior_token_indices,
            evidence_branch_ids=evidence_branch_ids,
            evidence_row_ids=evidence_row_ids,
            mix_req_ids=mix_req_ids,
            mix_branch_ids=mix_branch_ids,
            mix_row_ids=mix_row_ids,
            log_posterior=log_posterior,
            is_final_shard=is_final,
            direct_mix=direct_mix,
            sample_on_completion=sample_on_completion,
            debug=group.debug,
        )
        self._berag_debug(
            group,
            "emit shard scheduled_branches=%s evidence_branches=%s "
            "mix_branches=%s reqs=%s evidence_rows=%s mix_rows=%s "
            "mixture_row=%s direct_mix=%s prior_reqs=%s final=%s sample=%s "
            "pending_finalize=%s completed_before=%s active=%s",
            scheduled_branch_ids,
            evidence_branch_ids,
            mix_branch_ids,
            scheduled_req_ids,
            evidence_row_ids,
            mix_row_ids,
            shard.mixture_row_id,
            direct_mix,
            prior_req_ids,
            is_final,
            sample_on_completion,
            group.pending_finalize,
            sorted(group.completed_branch_ids),
            sorted(group.active_branch_ids),
        )
        self._write_berag_group_trace(
            group,
            "schedule_shard",
            scheduled_req_ids=scheduled_req_ids,
            scheduled_branch_ids=scheduled_branch_ids,
            evidence_branch_ids=evidence_branch_ids,
            evidence_row_ids=evidence_row_ids,
            mix_req_ids=mix_req_ids,
            mix_branch_ids=mix_branch_ids,
            mix_row_ids=mix_row_ids,
            mixture_row_id=shard.mixture_row_id,
            direct_mix=direct_mix,
            prior_req_ids=prior_req_ids,
            prior_branch_ids=prior_branch_ids,
            prior_token_indices=prior_token_indices,
            is_final_shard=is_final,
            sample_on_completion=sample_on_completion,
        )
        return shard

    def _schedule_berag_running_request(
        self,
        request: Request,
        token_budget: int,
        scheduled_running_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
        reserved_mixture_groups: set[str],
        reserved_branch_rows: set[tuple[str, int]],
    ) -> int:
        if request.status != RequestStatus.RUNNING:
            return token_budget
        if self.current_step < request.next_decode_eligible_step:
            return token_budget

        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        num_new_tokens = min(num_new_tokens, token_budget)
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len
            - request.num_computed_tokens
            - self.num_sampled_tokens_per_step,
        )
        if num_new_tokens <= 0:
            return token_budget

        if not self._try_reserve_berag_rows(
            request,
            request.num_computed_tokens,
            num_new_tokens,
            reserved_mixture_groups,
            reserved_branch_rows,
        ):
            return token_budget

        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )
        if new_blocks is None:
            return token_budget

        scheduled_running_reqs.append(request)
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        return token_budget - num_new_tokens

    def _schedule_berag_waiting_request(
        self,
        request: Request,
        token_budget: int,
        scheduled_timestamp: float,
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
        reserved_mixture_groups: set[str],
        reserved_branch_rows: set[tuple[str, int]],
    ) -> int:
        if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            return token_budget
        if len(self.running) == self.max_num_running_reqs:
            return token_budget

        if request.num_computed_tokens == 0:
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            if self.kv_cache_manager.log_stats:
                assert self.kv_cache_manager.prefix_cache_stats is not None
                self.kv_cache_manager.prefix_cache_stats.record(
                    num_tokens=request.num_tokens,
                    num_hits=num_new_local_computed_tokens,
                    preempted=request.num_preemptions > 0,
                )
            num_computed_tokens = num_new_local_computed_tokens
            if request.prefill_stats is not None:
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=num_new_local_computed_tokens,
                    num_external_cached_tokens=0,
                )
        else:
            new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        num_new_tokens = request.num_tokens - num_computed_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        if (
            not self.scheduler_config.enable_chunked_prefill
            and num_new_tokens > token_budget
        ):
            return token_budget
        num_new_tokens = min(num_new_tokens, token_budget)
        if num_new_tokens <= 0:
            return token_budget

        if not self._try_reserve_berag_rows(
            request,
            num_computed_tokens,
            num_new_tokens,
            reserved_mixture_groups,
            reserved_branch_rows,
        ):
            return token_budget

        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=self.num_lookahead_tokens,
            has_scheduled_reqs=bool(self.running),
        )
        if new_blocks is None:
            return token_budget

        self._remove_berag_waiting_request(request)
        self.running.append(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
        if request.status == RequestStatus.WAITING:
            scheduled_new_reqs.append(request)
        else:
            scheduled_resumed_reqs.append(request)

        req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(
            request.request_id
        )
        num_scheduled_tokens[request.request_id] = num_new_tokens
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        if num_computed_tokens + num_new_tokens < request.num_tokens:
            self._inflight_prefills.add(request)
        return token_budget - num_new_tokens

    def _make_scheduler_output_from_parts(
        self,
        *,
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        preempted_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        scheduled_encoder_inputs: dict[str, list[int]],
        scheduled_berag_shards: list[ScheduledBeragShard] | None,
    ) -> SchedulerOutput:
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            scheduled_berag_shards=scheduled_berag_shards,
            berag_release_rows=self._take_berag_release_rows(),
            berag_committed_tokens=self._take_berag_committed_tokens(),
        )
        if self.connector is not None:
            scheduler_output.kv_connector_metadata = self._build_kv_connector_meta(
                self.connector, scheduler_output
            )
        if self.ec_connector is not None:
            scheduler_output.ec_connector_metadata = (
                self.ec_connector.build_connector_meta(scheduler_output)
            )
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    @staticmethod
    def _berag_progress_error(reasons: dict[str, str]) -> RuntimeError:
        details = "; ".join(
            f"{group_id}: {reason}" for group_id, reason in reasons.items()
        )
        return RuntimeError(
            "No BERAG shard can be scheduled. BERAG needs at least one branch "
            "chunk or one deferred finalize shard to fit the current sequence, "
            "token, KV-cache, encoder, and accumulator-row budgets. Increase "
            "max_num_seqs, max_num_batched_tokens, max_num_encoder_input_tokens, "
            "or num_accumulator_rows; reduce K/document length; or enable "
            "chunked prefill for this workload. "
            f"Blocked groups: {details}."
        )

    def _schedule_berag(self, throttle_prefills: bool = False) -> SchedulerOutput:
        del throttle_prefills
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        scheduled_timestamp = time.monotonic()
        self.kv_cache_manager.new_step_starts()
        scheduled_berag_shards = self._collect_ready_berag_finalizes()
        virtual_running_reqs = len(self.running)
        blocked_berag_groups: dict[str, str] = {}

        if self._pause_state != PauseState.PAUSED_ALL:
            for group in self._berag_ordered_groups_for_work():
                if token_budget <= 0:
                    blocked_berag_groups[group.group_id] = (
                        "token budget is exhausted"
                    )
                    break
                pending_branch_ids = sorted(
                    group.active_branch_ids - group.completed_branch_ids
                )
                if not pending_branch_ids:
                    continue
                candidate_plans: list[BeragBranchSchedulePlan] = []
                candidate_budget = token_budget
                candidate_running_reqs = virtual_running_reqs
                for branch_id in pending_branch_ids:
                    if candidate_budget <= 0:
                        break
                    req_id = group.child_request_ids[branch_id]
                    request = self.requests.get(req_id)
                    if request is None or request.is_finished():
                        continue
                    plan = self._plan_berag_branch_request(
                        request, candidate_budget, candidate_running_reqs
                    )
                    if plan is None:
                        continue
                    candidate_plans.append(plan)
                    candidate_budget -= plan.num_new_tokens
                    if plan.was_waiting:
                        candidate_running_reqs += 1

                if not candidate_plans:
                    blocked_berag_groups[group.group_id] = (
                        "no branch could be planned within the current token/"
                        "sequence/KV budget"
                    )
                    continue

                direct_mix_candidate = self._berag_direct_mix_candidate(
                    group, candidate_plans
                )
                while candidate_plans and not direct_mix_candidate:
                    rows_needed = self._berag_rows_needed_for_plans(
                        group, candidate_plans
                    )
                    if rows_needed <= self.berag_row_allocator.free_count:
                        break
                    dropped = candidate_plans.pop()
                    self._berag_debug(
                        group,
                        "row backpressure dropped branch=%d rows_needed=%d "
                        "free=%d remaining_branches=%s",
                        dropped.branch_id,
                        rows_needed,
                        self.berag_row_allocator.free_count,
                        [plan.branch_id for plan in candidate_plans],
                    )
                    direct_mix_candidate = self._berag_direct_mix_candidate(
                        group, candidate_plans
                    )

                if not candidate_plans:
                    blocked_berag_groups[group.group_id] = (
                        "accumulator row budget cannot fit any "
                        "evidence-producing branch in this shard"
                    )
                    continue

                if not direct_mix_candidate:
                    rows_needed = self._berag_rows_needed_for_plans(
                        group, candidate_plans
                    )
                    if rows_needed > self.berag_row_allocator.free_count:
                        blocked_berag_groups[group.group_id] = (
                            f"accumulator path needs {rows_needed} rows but "
                            f"only {self.berag_row_allocator.free_count} "
                            "are free"
                        )
                        continue

                committed_plans: list[BeragBranchSchedulePlan] = []
                for plan in candidate_plans:
                    if token_budget < plan.num_new_tokens:
                        blocked_berag_groups[group.group_id] = (
                            "token budget was exhausted before committing all "
                            "planned branches"
                        )
                        break
                    if plan.was_waiting and virtual_running_reqs >= (
                        self.max_num_running_reqs
                    ):
                        blocked_berag_groups[group.group_id] = (
                            "sequence budget was exhausted before committing "
                            "all planned branches"
                        )
                        break
                    committed, encoder_compute_budget = (
                        self._commit_berag_branch_plan(
                            plan,
                            scheduled_timestamp,
                            scheduled_new_reqs,
                            scheduled_resumed_reqs,
                            scheduled_running_reqs,
                            req_to_new_blocks,
                            num_scheduled_tokens,
                            scheduled_encoder_inputs,
                            encoder_compute_budget,
                        )
                    )
                    if not committed:
                        blocked_berag_groups[group.group_id] = (
                            "KV cache or encoder scheduling rejected branch "
                            f"{plan.branch_id}"
                        )
                        break
                    committed_plans.append(plan)
                    token_budget -= plan.num_new_tokens
                    if plan.was_waiting:
                        virtual_running_reqs += 1

                if not committed_plans:
                    continue
                shard = self._make_berag_shard_from_committed_plans(
                    group, committed_plans
                )
                if shard is None:
                    raise RuntimeError(
                        "BERAG row reservation invariant violated after KV "
                        f"allocation: group={group.group_id}, "
                        f"branches={[p.branch_id for p in committed_plans]}, "
                        f"free_rows={self.berag_row_allocator.free_count}."
                    )
                scheduled_berag_shards.append(shard)
                blocked_berag_groups.pop(group.group_id, None)

        if (
            self._pause_state != PauseState.PAUSED_ALL
            and not num_scheduled_tokens
            and not scheduled_berag_shards
            and blocked_berag_groups
        ):
            raise self._berag_progress_error(blocked_berag_groups)

        assert sum(num_scheduled_tokens.values()) <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        return self._make_scheduler_output_from_parts(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_resumed_reqs=scheduled_resumed_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            preempted_reqs=preempted_reqs,
            req_to_new_blocks=req_to_new_blocks,
            num_scheduled_tokens=num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_berag_shards=scheduled_berag_shards or None,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        if self.berag_groups:
            return self._schedule_berag(throttle_prefills)
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        berag_reserved_mixture_groups: set[str] = set()
        berag_reserved_branch_rows: set[tuple[str, int]] = set()
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if not self._berag_group_is_ready_to_schedule(request):
                req_index += 1
                continue

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            if not self._try_reserve_berag_rows(
                request,
                request.num_computed_tokens,
                num_new_tokens,
                berag_reserved_mixture_groups,
                berag_reserved_branch_rows,
            ):
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                if not self._berag_group_is_ready_to_schedule(request):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                num_uncached_common_prefix_tokens = 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    if (
                        self.connector is not None
                        and self.has_mamba_layers
                        and isinstance(
                            self.kv_cache_manager.coordinator,
                            HybridKVCacheCoordinator,
                        )
                    ):
                        computed, per_group_hits = (
                            self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                                request.block_hashes,
                                request.num_tokens - 1,
                            )
                        )
                        new_computed_blocks = (
                            self.kv_cache_manager.create_kv_cache_blocks(computed)
                        )
                        # NOTE(ZhanqiuHu): For Mamba hybrid models,
                        # num_new_local_computed_tokens should be the FA hit
                        # length. This value is passed to the connector's
                        # get_num_new_matched_tokens which computes:
                        # external = total - local_computed.
                        # Using the FA hit skips re-transferring FA blocks
                        # already cached on D-side. The Mamba state (always
                        # the last block) is transferred unconditionally by
                        # _apply_prefix_caching in nixl/worker.py.
                        num_new_local_computed_tokens = max(per_group_hits)
                        if self.kv_cache_manager.log_stats:
                            assert self.kv_cache_manager.prefix_cache_stats is not None
                            self.kv_cache_manager.prefix_cache_stats.record(
                                num_tokens=request.num_tokens,
                                num_hits=num_new_local_computed_tokens,
                                preempted=request.num_preemptions > 0,
                            )
                    else:
                        new_computed_blocks, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request)
                        )

                    # In case of hybrid models, obtain hint for Marconi-style APC logic
                    if self.has_mamba_layers:
                        num_uncached_common_prefix_tokens = getattr(
                            self.kv_cache_manager.coordinator,
                            "num_uncached_common_prefix_tokens",
                            0,
                        )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Skip request with pending mm encoding prefetches
                    if (
                        self.ec_connector is not None
                        and request.mm_features
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens
                        )
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats is not None:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and request.num_computed_tokens == 0:
                    # DP prefill balancing: async KV loads (the branch above) are
                    # allowed to start even on throttled steps, but committing new
                    # prefill compute is deferred to a cadence-aligned step.
                    break
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                        num_uncached_common_prefix_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                if not self._try_reserve_berag_rows(
                    request,
                    num_computed_tokens,
                    num_new_tokens,
                    berag_reserved_mixture_groups,
                    berag_reserved_branch_rows,
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                limit_lookahead_tokens = load_kv_async and self.use_eagle
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            scheduled_berag_shards=self._build_scheduled_berag_shards(
                num_scheduled_tokens
            ),
            berag_release_rows=self._take_berag_release_rows(),
            berag_committed_tokens=self._take_berag_committed_tokens(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        self._check_berag_row_telemetry(model_runner_output)
        stopped_running_reqs: set[Request] = self._update_berag_from_output(
            model_runner_output, outputs
        )
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's +1 read has also passed it, mirroring
        # the shift the encoder scheduling path applies.
        spec_lookahead = 1 if self.use_eagle else 0

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        if request.berag_child is None and self.berag_groups:
            raise ValueError("Ordinary requests are not supported in BERAG mode.")
        if request.berag_child is not None and any(
            req.berag_child is None for req in self.requests.values()
        ):
            raise ValueError("BERAG requests cannot be mixed with ordinary requests.")

        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)
            if request.berag_child is not None:
                self._register_berag_child(request)

    def _register_berag_child(self, request: Request) -> None:
        meta = request.berag_child
        assert meta is not None
        group = self.berag_groups.get(meta.group_id)
        if group is None:
            group = BeragGroupState(
                group_id=meta.group_id,
                parent_request_id=meta.parent_request_id,
                num_branches=meta.num_branches,
                pruning_top_p=meta.pruning_top_p,
            )
            self.berag_groups[meta.group_id] = group
            self.berag_group_order.append(meta.group_id)
        group.register_child(request)
        if self.berag_config.prior_mode == "uniform":
            group.prior_scores[meta.branch_id] = 0.0
        self._berag_debug(
            group,
            "registered child req=%s branch=%d/%d prior_token_index=%d "
            "children=%d",
            request.request_id,
            meta.branch_id,
            meta.num_branches,
            meta.prior_token_index,
            len(group.child_request_ids),
        )

    def _finish_berag_child_from_abort(self, request: Request) -> None:
        meta = request.berag_child
        if meta is None:
            return
        group = self.berag_groups.get(meta.group_id)
        if group is None:
            return
        group.active_branch_ids.discard(meta.branch_id)
        group.log_posterior.pop(meta.branch_id, None)
        if group.active_branch_ids:
            return
        self._release_berag_step_rows(group)
        self.berag_groups.pop(meta.group_id, None)
        try:
            self.berag_group_order.remove(meta.group_id)
        except ValueError:
            pass

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)
            self._finish_berag_child_from_abort(request)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Entries are appended with monotonically non-decreasing fences, so
        stop at the first one that is still pending.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # For async scheduling, any output frames already in flight at
                # preemption time are now stale and must be discarded when they
                # return. num_output_placeholders is exactly that count: 0 if
                # the engine has drained (e.g. pause_generation(keep) waited
                # for idle), 1 for vanilla async mid-step, or 1 + spec/PP frames
                # otherwise.
                request.async_tokens_to_discard = request.num_output_placeholders
                request.num_output_placeholders = 0

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_computed_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
