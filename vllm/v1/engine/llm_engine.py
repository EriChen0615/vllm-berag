# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from typing import Any

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.berag import BeragChildMetadata, BeragParams
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_stats = log_stats

        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

            # Capture the model while reachable so the finalizer can drop the
            # bytecode hooks pinning it (frees GPU memory on engine deletion).
            model = self._get_driver_model_for_cleanup()
            if model is not None:
                self._finalizer = weakref.finalize(
                    self, LLMEngine._cleanup_instance_caches, model
                )

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            self.dp_group = get_dp_group().cpu_group

        self._berag_mode_active = False
        self._berag_child_ids_by_parent_id: dict[str, list[str]] = {}
        self.reset_benchmark_scheduler_stats()

        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    def reset_benchmark_scheduler_stats(self) -> None:
        self._benchmark_scheduler_stats: dict[str, int | float] = {
            "scheduler_steps": 0,
            "gpu_kv_cache_usage_final": 0.0,
            "gpu_kv_cache_usage_peak": 0.0,
            "prefix_cache_requests": 0,
            "prefix_cache_queries": 0,
            "prefix_cache_hits": 0,
            "connector_prefix_cache_requests": 0,
            "connector_prefix_cache_queries": 0,
            "connector_prefix_cache_hits": 0,
        }

    def _record_benchmark_scheduler_stats(
        self, scheduler_stats: Any | None
    ) -> None:
        if scheduler_stats is None:
            return
        stats = self._benchmark_scheduler_stats
        stats["scheduler_steps"] += 1
        stats["gpu_kv_cache_usage_final"] = scheduler_stats.kv_cache_usage
        stats["gpu_kv_cache_usage_peak"] = max(
            float(stats["gpu_kv_cache_usage_peak"]),
            scheduler_stats.kv_cache_usage,
        )

        prefix_stats = scheduler_stats.prefix_cache_stats
        if prefix_stats.reset:
            stats["prefix_cache_requests"] = 0
            stats["prefix_cache_queries"] = 0
            stats["prefix_cache_hits"] = 0
        stats["prefix_cache_requests"] += prefix_stats.requests
        stats["prefix_cache_queries"] += prefix_stats.queries
        stats["prefix_cache_hits"] += prefix_stats.hits

        connector_stats = scheduler_stats.connector_prefix_cache_stats
        if connector_stats is not None:
            if connector_stats.reset:
                stats["connector_prefix_cache_requests"] = 0
                stats["connector_prefix_cache_queries"] = 0
                stats["connector_prefix_cache_hits"] = 0
            stats["connector_prefix_cache_requests"] += connector_stats.requests
            stats["connector_prefix_cache_queries"] += connector_stats.queries
            stats["connector_prefix_cache_hits"] += connector_stats.hits

    def get_benchmark_scheduler_stats(self) -> dict[str, int | float]:
        stats = dict(self._benchmark_scheduler_stats)
        prefix_queries = float(stats["prefix_cache_queries"])
        prefix_hits = float(stats["prefix_cache_hits"])
        connector_queries = float(stats["connector_prefix_cache_queries"])
        connector_hits = float(stats["connector_prefix_cache_hits"])
        stats["gpu_kv_cache_usage_final_pct"] = (
            float(stats["gpu_kv_cache_usage_final"]) * 100
        )
        stats["gpu_kv_cache_usage_peak_pct"] = (
            float(stats["gpu_kv_cache_usage_peak"]) * 100
        )
        stats["prefix_cache_hit_rate"] = (
            prefix_hits / prefix_queries if prefix_queries > 0 else 0.0
        )
        stats["prefix_cache_hit_rate_pct"] = (
            float(stats["prefix_cache_hit_rate"]) * 100
        )
        stats["connector_prefix_cache_hit_rate"] = (
            connector_hits / connector_queries if connector_queries > 0 else 0.0
        )
        stats["connector_prefix_cache_hit_rate_pct"] = (
            float(stats["connector_prefix_cache_hit_rate"]) * 100
        )
        return stats

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        request_ids = self._expand_berag_abort_request_ids(request_ids)
        self.engine_core.abort_requests(request_ids)

    def _expand_berag_abort_request_ids(self, request_ids: list[str]) -> list[str]:
        expanded_request_ids: list[str] = []
        for request_id in request_ids:
            expanded_request_ids.append(request_id)
            expanded_request_ids.extend(
                self._berag_child_ids_by_parent_id.pop(request_id, [])
            )
        return expanded_request_ids

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> str:
        if self._berag_mode_active:
            raise ValueError("Ordinary requests are not supported in BERAG mode.")

        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return req_id

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

        return req_id

    def add_berag_request(
        self,
        request_id: str,
        shared_prefix: str | PromptType,
        documents: list[str],
        suffix: str,
        sampling_params: SamplingParams,
        berag_params: BeragParams | None = None,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        debug: bool = False,
    ) -> str:
        berag_params = berag_params or BeragParams()
        self._validate_berag_request(documents, sampling_params, berag_params)
        explicit_prior_indices = berag_params.prior_token_indices
        if explicit_prior_indices is not None and len(explicit_prior_indices) != len(
            documents
        ):
            raise ValueError(
                "BeragParams.prior_token_indices must have one entry per document."
            )

        mode_was_active = self._berag_mode_active
        self._berag_mode_active = True
        parent_params = copy(sampling_params)
        parent_params.output_kind = RequestOutputKind.FINAL_ONLY
        shared_prefix_text, shared_prompt = self._prepare_berag_shared_prefix(
            request_id,
            shared_prefix,
        )
        parent_prompt = self._berag_prompt_with_text(
            shared_prompt,
            f"{shared_prefix_text}{suffix}",
        )

        child_request_ids: list[str] = []
        parent_id: str | None = None
        try:
            parent_req = self.input_processor.process_inputs(
                request_id,
                parent_prompt,
                parent_params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                skip_mm_cache=True,
            )
            self.input_processor.assign_request_id(parent_req)
            parent_id = parent_req.request_id
            parent_prompt_text, _, _ = extract_prompt_components(
                self.model_config, parent_prompt
            )
            self.output_processor.add_request(parent_req, parent_prompt_text, None, 0)
            parent_prompt_len = len(parent_req.prompt_token_ids or [])
            if debug:
                logger.info(
                    "[BERAG debug] admission parent=%s external=%s branches=%d "
                    "parent_prompt_len=%d max_tokens=%s pruning_top_p=%s",
                    parent_id,
                    parent_req.external_req_id,
                    len(documents),
                    parent_prompt_len,
                    sampling_params.max_tokens,
                    berag_params.pruning_top_p,
                )

            for branch_id, document in enumerate(documents):
                child_prompt = self._berag_prompt_with_text(
                    shared_prompt,
                    f"{shared_prefix_text}{document}{suffix}",
                )
                child_params = copy(sampling_params)
                child_params.n = 1
                child_params.output_kind = RequestOutputKind.FINAL_ONLY
                child_req_id = f"{parent_id}:berag:{branch_id}"
                child_req = self.input_processor.process_inputs(
                    child_req_id,
                    child_prompt,
                    child_params,
                    supported_tasks=self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                )
                assert child_req.prompt_token_ids is not None
                prior_index = self._resolve_berag_prior_index(
                    len(child_req.prompt_token_ids),
                    explicit_prior_indices[branch_id]
                    if explicit_prior_indices is not None
                    else self.vllm_config.berag_config.default_prior_token_offset,
                )
                max_tokens = child_params.max_tokens or 0
                if len(child_req.prompt_token_ids) + max_tokens > (
                    self.model_config.max_model_len
                ):
                    raise ValueError(
                        "BERAG child prompt plus max_tokens exceeds max_model_len: "
                        f"branch_id={branch_id}, prompt_len="
                        f"{len(child_req.prompt_token_ids)}, "
                        f"max_tokens={max_tokens}, "
                        f"max_model_len={self.model_config.max_model_len}."
                    )
                child_req.berag_child = BeragChildMetadata(
                    group_id=parent_id,
                    parent_request_id=parent_id,
                    branch_id=branch_id,
                    num_branches=len(documents),
                    parent_prompt_len=parent_prompt_len,
                    prior_token_index=prior_index,
                    pruning_top_p=berag_params.pruning_top_p,
                    debug=debug,
                )
                child_req.external_req_id = child_req.request_id
                if debug:
                    logger.info(
                        "[BERAG debug] admission child=%s branch=%d/%d "
                        "prompt_len=%d prior_token_index=%d max_tokens=%s",
                        child_req.request_id,
                        branch_id,
                        len(documents),
                        len(child_req.prompt_token_ids),
                        prior_index,
                        child_params.max_tokens,
                    )
                self.engine_core.add_request(child_req)
                child_request_ids.append(child_req.request_id)
        except Exception:
            if parent_id is not None:
                self.output_processor.abort_requests([parent_id], internal=True)
            if child_request_ids:
                self.engine_core.abort_requests(child_request_ids)
            if parent_id is not None:
                self._berag_child_ids_by_parent_id.pop(parent_id, None)
            if not mode_was_active:
                self._berag_mode_active = False
            raise

        assert parent_id is not None
        self._berag_child_ids_by_parent_id[parent_id] = child_request_ids
        return parent_id

    @staticmethod
    def _berag_extract_text_prefix(shared_prefix: str | PromptType) -> str:
        if isinstance(shared_prefix, str):
            return shared_prefix
        if isinstance(shared_prefix, Mapping):
            prompt = shared_prefix.get("prompt")
            if isinstance(prompt, str):
                return prompt
        raise TypeError(
            "BERAG shared_prefix must be a string or a text PromptType "
            "dictionary with a string 'prompt' field."
        )

    @staticmethod
    def _berag_mm_item_count(value: object) -> int:
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return len(value)
        return 1

    @classmethod
    def _berag_make_shared_mm_uuids(
        cls,
        parent_request_id: str,
        multi_modal_data: Mapping[str, object],
    ) -> dict[str, list[str]]:
        return {
            modality: [
                f"{parent_request_id}:berag:mm:{modality}:{index}"
                for index in range(cls._berag_mm_item_count(items))
            ]
            for modality, items in multi_modal_data.items()
        }

    @classmethod
    def _prepare_berag_shared_prefix(
        cls,
        parent_request_id: str,
        shared_prefix: str | PromptType,
    ) -> tuple[str, str | PromptType]:
        shared_prefix_text = cls._berag_extract_text_prefix(shared_prefix)
        if isinstance(shared_prefix, str):
            return shared_prefix_text, shared_prefix

        shared_prompt = dict(shared_prefix)
        multi_modal_data = shared_prompt.get("multi_modal_data")
        if (
            isinstance(multi_modal_data, Mapping)
            and "multi_modal_uuids" not in shared_prompt
        ):
            shared_prompt["multi_modal_uuids"] = cls._berag_make_shared_mm_uuids(
                parent_request_id,
                multi_modal_data,
            )
        return shared_prefix_text, shared_prompt  # type: ignore[return-value]

    @staticmethod
    def _berag_prompt_with_text(
        shared_prompt: str | PromptType,
        text: str,
    ) -> str | PromptType:
        if isinstance(shared_prompt, str):
            return text
        prompt = dict(shared_prompt)
        prompt["prompt"] = text
        return prompt  # type: ignore[return-value]

    def _validate_berag_request(
        self,
        documents: list[str],
        sampling_params: SamplingParams,
        berag_params: BeragParams,
    ) -> None:
        if not documents:
            raise ValueError("BERAG requires at least one document branch.")
        if not (0.0 < berag_params.pruning_top_p <= 1.0):
            raise ValueError("BeragParams.pruning_top_p must be in (0, 1].")
        berag_config = self.vllm_config.berag_config
        if berag_config.prior_mode == "module" and (
            not berag_config.prior_module_cls
            or not berag_config.prior_module_weights_path
        ):
            raise ValueError(
                "BERAG requires berag_prior_module_cls and "
                "berag_prior_module_weights_path in module prior mode."
            )
        parallel_config = self.vllm_config.parallel_config
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.data_parallel_size != 1
        ):
            raise ValueError("BERAG currently supports only single-GPU execution.")
        if self.vllm_config.scheduler_config.async_scheduling:
            raise ValueError("BERAG does not support async scheduling.")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("BERAG does not support speculative decoding.")
        if sampling_params.n != 1:
            raise ValueError("BERAG requires SamplingParams.n == 1.")
        if sampling_params.structured_outputs is not None:
            raise ValueError("BERAG does not support structured-output grammar.")
        if sampling_params.repetition_penalty != 1.0:
            raise ValueError(
                "BERAG does not support repetition_penalty because parent-level "
                "prompt masking is not represented by child worker rows."
            )
        if sampling_params.min_tokens:
            raise ValueError(
                "BERAG does not support min_tokens because parent-level "
                "minimum-length masking is not represented by child worker rows."
            )
        if sampling_params.bad_words_token_ids:
            raise ValueError(
                "BERAG does not support bad_words because parent-level bad-word "
                "matching is not represented by child worker rows."
            )

    @staticmethod
    def _resolve_berag_prior_index(prompt_len: int, index: int) -> int:
        resolved = prompt_len + index if index < 0 else index
        if resolved < 0 or resolved >= prompt_len:
            raise ValueError(
                f"BERAG prior_token_index {index} is outside prompt length "
                f"{prompt_len}."
            )
        return resolved

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)
            self._record_benchmark_scheduler_stats(outputs.scheduler_stats)
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self._berag_child_ids_by_parent_id.pop(output.request_id, None)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            reqs_to_abort = self._expand_berag_abort_request_ids(
                processed_outputs.reqs_to_abort
            )
            self.engine_core.abort_requests(reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                self.logger_manager is not None
                and outputs.scheduler_stats is not None
                and len(outputs.outputs) > 0
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self, profile_prefix: str | None = None):
        self.engine_core.profile(True, profile_prefix)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.renderer.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        if level >= 1:
            self.renderer.clear_mm_cache()
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def _get_driver_model_for_cleanup(self) -> nn.Module | None:
        driver_worker = getattr(self.model_executor, "driver_worker", None)
        model_runner = getattr(driver_worker, "model_runner", None)
        return getattr(model_runner, "model", None)

    @staticmethod
    def _cleanup_instance_caches(model) -> None:
        """Remove the bytecode hooks that pin the compiled model."""
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        for module in model.modules():
            if isinstance(module, TorchCompileWithNoGuardsWrapper):
                module.cleanup()

    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
