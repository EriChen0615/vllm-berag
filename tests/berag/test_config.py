# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.berag import BeragParams
from vllm.config.berag import BeragConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine


def make_fake_engine_config(
    *,
    prior_mode: str = "module",
    prior_module_cls: str | None = "tests.berag.prior_fixtures.TinyPrior",
    prior_module_weights_path: str | None = "/tmp/prior.pt",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    async_scheduling: bool = False,
    speculative_config: object | None = None,
):
    return SimpleNamespace(
        berag_config=BeragConfig(
            prior_mode=prior_mode,
            prior_module_cls=prior_module_cls,
            prior_module_weights_path=prior_module_weights_path,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative_config,
    )


def validate(
    vllm_config,
    sampling_params: SamplingParams | None = None,
    berag_params: BeragParams | None = None,
) -> None:
    fake_engine = SimpleNamespace(vllm_config=vllm_config)
    LLMEngine._validate_berag_request(
        fake_engine,
        ["doc"],
        sampling_params or SamplingParams(max_tokens=4),
        berag_params or BeragParams(),
    )


def test_berag_config_enabled_requires_prior_fields_for_validation():
    config = BeragConfig()
    assert not config.enabled
    assert BeragConfig(prior_mode="uniform").enabled

    validate(
        make_fake_engine_config(
            prior_mode="uniform",
            prior_module_cls=None,
            prior_module_weights_path=None,
        )
    )

    with pytest.raises(ValueError, match="prior_module_cls"):
        validate(make_fake_engine_config(prior_module_cls=None))

    with pytest.raises(ValueError, match="prior_module_weights_path"):
        validate(make_fake_engine_config(prior_module_weights_path=None))


def test_prior_index_resolution():
    assert LLMEngine._resolve_berag_prior_index(10, -4) == 6
    assert LLMEngine._resolve_berag_prior_index(10, 3) == 3

    with pytest.raises(ValueError, match="outside prompt length"):
        LLMEngine._resolve_berag_prior_index(3, -4)

    with pytest.raises(ValueError, match="outside prompt length"):
        LLMEngine._resolve_berag_prior_index(3, 3)


def test_berag_validation_rejects_unsupported_engine_modes():
    with pytest.raises(ValueError, match="single-GPU"):
        validate(make_fake_engine_config(tensor_parallel_size=2))

    with pytest.raises(ValueError, match="async scheduling"):
        validate(make_fake_engine_config(async_scheduling=True))

    with pytest.raises(ValueError, match="speculative"):
        validate(make_fake_engine_config(speculative_config=object()))


def test_berag_validation_rejects_unsupported_sampling_params():
    config = make_fake_engine_config()

    with pytest.raises(ValueError, match="SamplingParams.n"):
        validate(config, SamplingParams(max_tokens=4, n=2))

    with pytest.raises(ValueError, match="repetition_penalty"):
        validate(config, SamplingParams(max_tokens=4, repetition_penalty=1.1))

    with pytest.raises(ValueError, match="min_tokens"):
        validate(config, SamplingParams(max_tokens=4, min_tokens=1))

    with pytest.raises(ValueError, match="pruning_top_p"):
        validate(config, berag_params=BeragParams(pruning_top_p=0.0))
