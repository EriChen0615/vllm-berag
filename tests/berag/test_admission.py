# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from itertools import count

import pytest

from tests.berag.test_config import make_fake_engine_config
from vllm.berag import BeragParams
from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine


class FakeInputProcessor:

    def __init__(self) -> None:
        self.prompts: dict[str, object] = {}
        self.skip_mm_cache: dict[str, bool] = {}

    def process_inputs(
        self,
        request_id,
        prompt,
        params,
        *,
        arrival_time=None,
        lora_request=None,
        trace_headers=None,
        priority=0,
        skip_mm_cache=False,
        **_,
    ):
        self.prompts[request_id] = prompt
        self.skip_mm_cache[request_id] = skip_mm_cache
        prompt_text = prompt["prompt"] if isinstance(prompt, Mapping) else prompt
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=list(range(len(prompt_text))),
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            arrival_time=arrival_time or 0.0,
            lora_request=lora_request,
            cache_salt=None,
            data_parallel_rank=None,
            trace_headers=trace_headers,
            priority=priority,
        )

    @staticmethod
    def assign_request_id(request) -> None:
        request.external_req_id = request.request_id


class FakeOutputProcessor:

    def __init__(self) -> None:
        self.added = []
        self.aborted = []

    def add_request(self, request, prompt_text, parent_req, index) -> None:
        self.added.append((request, prompt_text, parent_req, index))

    def abort_requests(self, request_ids, internal=False):
        self.aborted.append((request_ids, internal))
        return request_ids


class FakeEngineCore:

    def __init__(self) -> None:
        self.added = []
        self.aborted = []

    def add_request(self, request) -> None:
        self.added.append(request)

    def abort_requests(self, request_ids) -> None:
        self.aborted.append(request_ids)


class FakeLLMEngine:

    def __init__(self, *, max_model_len: int = 128) -> None:
        self.vllm_config = make_fake_engine_config()
        self.model_config = type(
            "FakeModelConfig",
            (),
            {
                "max_model_len": max_model_len,
                "is_encoder_decoder": False,
            },
        )()
        self.input_processor = FakeInputProcessor()
        self.output_processor = FakeOutputProcessor()
        self.engine_core = FakeEngineCore()
        self._berag_mode_active = False
        self._berag_child_ids_by_parent_id = {}

    @staticmethod
    def get_supported_tasks():
        return ("generate",)

    def _validate_berag_request(self, documents, sampling_params, berag_params):
        return LLMEngine._validate_berag_request(
            self,
            documents,
            sampling_params,
            berag_params,
        )

    @staticmethod
    def _resolve_berag_prior_index(prompt_len: int, index: int) -> int:
        return LLMEngine._resolve_berag_prior_index(prompt_len, index)

    @staticmethod
    def _prepare_berag_shared_prefix(parent_request_id, shared_prefix):
        return LLMEngine._prepare_berag_shared_prefix(
            parent_request_id,
            shared_prefix,
        )

    @staticmethod
    def _berag_prompt_with_text(shared_prompt, text):
        return LLMEngine._berag_prompt_with_text(shared_prompt, text)


def make_fake_engine(*, max_model_len: int = 128):
    return FakeLLMEngine(max_model_len=max_model_len)


def test_add_berag_request_expands_parent_into_internal_children():
    engine = make_fake_engine()
    sampling_params = SamplingParams(max_tokens=2)

    parent_id = LLMEngine.add_berag_request(
        engine,
        "parent",
        "question: ",
        ["doc-a ", "doc-b "],
        " answer:",
        sampling_params,
        berag_params=BeragParams(
            pruning_top_p=0.75,
            prior_token_indices=[3, 5],
        ),
        debug=True,
    )

    assert parent_id == "parent"
    assert engine._berag_mode_active
    assert engine._berag_child_ids_by_parent_id == {
        "parent": ["parent:berag:0", "parent:berag:1"]
    }
    assert [req.request_id for req in engine.engine_core.added] == [
        "parent:berag:0",
        "parent:berag:1",
    ]
    assert len(engine.output_processor.added) == 1
    parent_req, parent_prompt_text, parent_state, parent_index = (
        engine.output_processor.added[0]
    )
    assert parent_req.request_id == "parent"
    assert parent_prompt_text == "question:  answer:"
    assert parent_state is None
    assert parent_index == 0

    child0, child1 = engine.engine_core.added
    assert engine.input_processor.prompts[child0.request_id] == (
        "question: doc-a  answer:"
    )
    assert engine.input_processor.prompts[child1.request_id] == (
        "question: doc-b  answer:"
    )
    assert child0.external_req_id == child0.request_id
    assert child1.external_req_id == child1.request_id
    assert child0.sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert child1.sampling_params.output_kind == RequestOutputKind.FINAL_ONLY

    assert child0.berag_child.group_id == "parent"
    assert child0.berag_child.parent_request_id == "parent"
    assert child0.berag_child.branch_id == 0
    assert child0.berag_child.num_branches == 2
    assert child0.berag_child.parent_prompt_len == len("question:  answer:")
    assert child0.berag_child.prior_token_index == 3
    assert child0.berag_child.pruning_top_p == 0.75
    assert child0.berag_child.debug

    assert child1.berag_child.branch_id == 1
    assert child1.berag_child.prior_token_index == 5


def test_add_berag_request_reuses_shared_multimodal_prompt_fields():
    engine = make_fake_engine()
    image = object()
    shared_prefix = {
        "prompt": "<image> question: ",
        "multi_modal_data": {"image": image},
        "multi_modal_uuids": {"image": ["query-image"]},
        "mm_processor_kwargs": {"do_resize": False},
    }

    LLMEngine.add_berag_request(
        engine,
        "parent",
        shared_prefix,
        ["doc-a ", "doc-b "],
        " answer:",
        SamplingParams(max_tokens=1),
    )

    parent_prompt = engine.input_processor.prompts["parent"]
    child0_prompt = engine.input_processor.prompts["parent:berag:0"]
    child1_prompt = engine.input_processor.prompts["parent:berag:1"]
    assert isinstance(parent_prompt, dict)
    assert isinstance(child0_prompt, dict)
    assert isinstance(child1_prompt, dict)
    assert parent_prompt["prompt"] == "<image> question:  answer:"
    assert child0_prompt["prompt"] == "<image> question: doc-a  answer:"
    assert child1_prompt["prompt"] == "<image> question: doc-b  answer:"
    assert child0_prompt["multi_modal_data"] is parent_prompt["multi_modal_data"]
    assert child1_prompt["multi_modal_data"] is parent_prompt["multi_modal_data"]
    assert child0_prompt["multi_modal_uuids"] is parent_prompt["multi_modal_uuids"]
    assert child1_prompt["multi_modal_uuids"] is parent_prompt["multi_modal_uuids"]
    assert parent_prompt["multi_modal_uuids"] == {"image": ["query-image"]}
    assert child0_prompt["mm_processor_kwargs"] == {"do_resize": False}
    assert engine.input_processor.skip_mm_cache == {
        "parent": True,
        "parent:berag:0": False,
        "parent:berag:1": False,
    }


def test_add_berag_request_generates_stable_shared_multimodal_uuids():
    engine = make_fake_engine()
    shared_prefix = {
        "prompt": "<image><image> question: ",
        "multi_modal_data": {"image": [object(), object()]},
    }

    LLMEngine.add_berag_request(
        engine,
        "parent",
        shared_prefix,
        ["doc-a ", "doc-b "],
        " answer:",
        SamplingParams(max_tokens=1),
    )

    parent_prompt = engine.input_processor.prompts["parent"]
    child0_prompt = engine.input_processor.prompts["parent:berag:0"]
    child1_prompt = engine.input_processor.prompts["parent:berag:1"]
    assert isinstance(parent_prompt, dict)
    assert isinstance(child0_prompt, dict)
    assert isinstance(child1_prompt, dict)
    assert parent_prompt["multi_modal_uuids"] == {
        "image": [
            "parent:berag:mm:image:0",
            "parent:berag:mm:image:1",
        ]
    }
    assert child0_prompt["multi_modal_uuids"] is parent_prompt["multi_modal_uuids"]
    assert child1_prompt["multi_modal_uuids"] is parent_prompt["multi_modal_uuids"]


def test_add_berag_request_cleans_up_parent_and_children_on_admission_failure():
    engine = make_fake_engine(max_model_len=4)
    engine.vllm_config.berag_config.default_prior_token_offset = -1

    with pytest.raises(ValueError, match="exceeds max_model_len"):
        LLMEngine.add_berag_request(
            engine,
            "parent",
            "",
            ["a", "document-too-long"],
            "",
            SamplingParams(max_tokens=1),
        )

    assert [req.request_id for req in engine.engine_core.added] == [
        "parent:berag:0"
    ]
    assert engine.output_processor.aborted == [(["parent"], True)]
    assert engine.engine_core.aborted == [["parent:berag:0"]]
    assert engine._berag_child_ids_by_parent_id == {}
    assert not engine._berag_mode_active


def test_add_berag_request_rejects_wrong_prior_index_count_before_admission():
    engine = make_fake_engine()

    with pytest.raises(ValueError, match="one entry per document"):
        LLMEngine.add_berag_request(
            engine,
            "parent",
            "prefix ",
            ["doc0", "doc1"],
            " suffix",
            SamplingParams(max_tokens=1),
            berag_params=BeragParams(prior_token_indices=[0]),
        )

    assert not engine.engine_core.added
    assert not engine.output_processor.added
    assert not engine._berag_mode_active


def test_abort_expands_parent_request_to_internal_children():
    engine = make_fake_engine()
    engine._berag_child_ids_by_parent_id = {
        "parent": ["parent:berag:0", "parent:berag:1"]
    }

    expanded = LLMEngine._expand_berag_abort_request_ids(
        engine,
        ["other", "parent"],
    )

    assert expanded == [
        "other",
        "parent",
        "parent:berag:0",
        "parent:berag:1",
    ]
    assert engine._berag_child_ids_by_parent_id == {}


class FakeOfflineEngine:

    def __init__(self) -> None:
        self.berag_calls = []
        self.aborted = []

    def add_berag_request(self, *args, **kwargs) -> None:
        self.berag_calls.append((args, kwargs))

    def abort_request(self, request_ids, internal=False) -> None:
        self.aborted.append((request_ids, internal))

    def reset_benchmark_scheduler_stats(self) -> None:
        pass

    def get_benchmark_scheduler_stats(self) -> dict:
        return {}


class FakeRequestOutput:

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id


class FakeOfflineLLM:

    def __init__(self) -> None:
        self.model_config = type(
            "FakeModelConfig",
            (),
            {"runner_type": "generate"},
        )()
        self.request_counter = count()
        self.llm_engine = FakeOfflineEngine()
        self.run_engine_calls = []
        self.outputs = None

    @staticmethod
    def get_default_sampling_params() -> SamplingParams:
        return SamplingParams(max_tokens=3)

    def _run_engine(self, **kwargs):
        self.run_engine_calls.append(kwargs)
        if self.outputs is not None:
            return self.outputs
        return [
            FakeRequestOutput(args[0])
            for args, _ in self.llm_engine.berag_calls
        ]


def test_generate_berag_uses_parent_request_id_and_final_only_sampling():
    llm = FakeOfflineLLM()
    sampling_params = SamplingParams(max_tokens=2)

    outputs = LLM.generate_berag(
        llm,
        "prefix ",
        ["doc"],
        " suffix",
        sampling_params,
        berag_params=BeragParams(pruning_top_p=0.9),
        request_id="req-1",
        use_tqdm=False,
        debug=True,
    )

    assert [output.request_id for output in outputs] == ["req-1"]
    assert sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    args, kwargs = llm.llm_engine.berag_calls[0]
    assert args[:5] == (
        "req-1",
        "prefix ",
        ["doc"],
        " suffix",
        sampling_params,
    )
    assert kwargs["berag_params"].pruning_top_p == 0.9
    assert kwargs["debug"]
    assert llm.run_engine_calls == [
        {
            "use_tqdm": False,
            "output_type": RequestOutput,
        }
    ]


def test_generate_berag_accepts_multimodal_prompt_prefix():
    llm = FakeOfflineLLM()
    shared_prefix = {
        "prompt": "<image> prefix ",
        "multi_modal_data": {"image": object()},
        "multi_modal_uuids": {"image": ["query-image"]},
    }

    outputs = LLM.generate_berag(
        llm,
        shared_prefix,
        ["doc"],
        " suffix",
        SamplingParams(max_tokens=2),
        request_id="req-1",
        use_tqdm=False,
    )

    assert [output.request_id for output in outputs] == ["req-1"]
    args, _ = llm.llm_engine.berag_calls[0]
    assert args[0] == "req-1"
    assert args[1] is shared_prefix
    assert args[2] == ["doc"]
    assert args[3] == " suffix"
    assert isinstance(args[4], SamplingParams)


def test_generate_berag_batches_parent_requests_before_engine_run():
    llm = FakeOfflineLLM()
    sampling_params = SamplingParams(max_tokens=2)
    llm.outputs = [
        FakeRequestOutput("req-b"),
        FakeRequestOutput("req-a"),
    ]

    outputs = LLM.generate_berag(
        llm,
        ["prefix-a ", "prefix-b "],
        [["doc-a0", "doc-a1"], ["doc-b0"]],
        [" suffix-a", " suffix-b"],
        sampling_params,
        berag_params=BeragParams(pruning_top_p=0.8),
        request_id=["req-a", "req-b"],
        priority=[3, 7],
        use_tqdm=False,
    )

    assert [output.request_id for output in outputs] == ["req-a", "req-b"]
    assert sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert len(llm.llm_engine.berag_calls) == 2
    first_args, first_kwargs = llm.llm_engine.berag_calls[0]
    second_args, second_kwargs = llm.llm_engine.berag_calls[1]
    assert first_args[:5] == (
        "req-a",
        "prefix-a ",
        ["doc-a0", "doc-a1"],
        " suffix-a",
        sampling_params,
    )
    assert second_args[:5] == (
        "req-b",
        "prefix-b ",
        ["doc-b0"],
        " suffix-b",
        sampling_params,
    )
    assert first_kwargs["berag_params"].pruning_top_p == 0.8
    assert second_kwargs["berag_params"].pruning_top_p == 0.8
    assert first_kwargs["priority"] == 3
    assert second_kwargs["priority"] == 7
    assert llm.run_engine_calls == [
        {
            "use_tqdm": False,
            "output_type": RequestOutput,
        }
    ]


def test_generate_berag_rejects_scalar_request_id_for_batch():
    llm = FakeOfflineLLM()

    with pytest.raises(ValueError, match="request_id to be a sequence"):
        LLM.generate_berag(
            llm,
            ["prefix-a ", "prefix-b "],
            [["doc-a"], ["doc-b"]],
            [" suffix-a", " suffix-b"],
            request_id="req",
        )


def test_generate_berag_rejects_duplicate_batch_request_ids():
    llm = FakeOfflineLLM()

    with pytest.raises(ValueError, match="request_id values must be unique"):
        LLM.generate_berag(
            llm,
            ["prefix-a ", "prefix-b "],
            [["doc-a"], ["doc-b"]],
            [" suffix-a", " suffix-b"],
            request_id=["req", "req"],
        )


def test_generate_berag_rejects_non_generate_runner():
    llm = FakeOfflineLLM()
    llm.model_config.runner_type = "pooling"

    with pytest.raises(ValueError, match="generative model"):
        LLM.generate_berag(llm, "prefix", ["doc"], "suffix")
