# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.cpu_test


def test_text_multimodal_preprocess_forwards_uuids_and_skip_mm_cache():
    input_preprocessor = InputPreprocessor.__new__(InputPreprocessor)
    input_preprocessor.model_config = type(
        "FakeModelConfig",
        (),
        {"is_encoder_decoder": False},
    )()
    calls = []
    image = object()

    def fake_process_multimodal(
        prompt,
        mm_data,
        mm_processor_kwargs=None,
        tokenization_kwargs=None,
        *,
        mm_uuids=None,
        skip_mm_cache=False,
    ):
        calls.append(
            {
                "prompt": prompt,
                "mm_data": mm_data,
                "mm_processor_kwargs": mm_processor_kwargs,
                "tokenization_kwargs": tokenization_kwargs,
                "mm_uuids": mm_uuids,
                "skip_mm_cache": skip_mm_cache,
            }
        )
        return {"type": "multimodal", "prompt_token_ids": [1, 2]}

    input_preprocessor._process_multimodal = fake_process_multimodal

    processed_inputs = InputPreprocessor.preprocess(
        input_preprocessor,
        {
            "prompt": "<image> describe",
            "multi_modal_data": {"image": image},
            "multi_modal_uuids": {"image": ["shared-image"]},
            "mm_processor_kwargs": {"do_resize": False},
        },
        tokenization_kwargs={"add_special_tokens": False},
        skip_mm_cache=True,
    )

    assert processed_inputs["prompt"] == "<image> describe"
    assert calls == [
        {
            "prompt": "<image> describe",
            "mm_data": {"image": image},
            "mm_processor_kwargs": {"do_resize": False},
            "tokenization_kwargs": {"add_special_tokens": False},
            "mm_uuids": {"image": ["shared-image"]},
            "skip_mm_cache": True,
        }
    ]


@pytest.mark.parametrize("model_id", ["facebook/chameleon-7b"])
@pytest.mark.parametrize("prompt", ["", {"prompt_token_ids": []}])
@pytest.mark.skip(
    reason=(
        "Applying huggingface processor on text inputs results in "
        "significant performance regression for multimodal models. "
        "See https://github.com/vllm-project/vllm/issues/26320"
    )
)
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    vllm_config = VllmConfig(model_config=model_config)
    input_preprocessor = InputPreprocessor(vllm_config)

    # HF processor adds sep token
    tokenizer = input_preprocessor.get_tokenizer()
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]
