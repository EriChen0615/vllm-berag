# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.berag import BeragParams

from .prior_fixtures import TinyPrior


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BERAG_E2E") != "1",
    reason="Set RUN_BERAG_E2E=1 to run the Qwen2.5 BERAG smoke test.",
)


def test_generate_berag_qwen_smoke(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Qwen BERAG smoke test requires CUDA.")

    prior = TinyPrior(hidden_size=896)
    prior_path = tmp_path / "tiny_prior.pt"
    torch.save(prior.state_dict(), prior_path)

    llm = LLM(
        model=os.environ.get("BERAG_E2E_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=96,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=False,
        berag_prior_module_cls="tests.berag.prior_fixtures.TinyPrior",
        berag_prior_module_weights_path=str(prior_path),
        berag_prior_module_kwargs={"hidden_size": 896},
    )

    outputs = llm.generate_berag(
        shared_prefix="Question: What color is the sky?\n",
        documents=[
            "Document: The daytime sky is usually blue.\n",
            "Document: Grass is often green.\n",
            "Document: Snow is often white.\n",
        ],
        suffix="Answer:",
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        berag_params=BeragParams(pruning_top_p=1.0),
        debug=True,
    )

    assert len(outputs) == 1
    assert outputs[0].request_id == "0"
    assert outputs[0].finished
    assert outputs[0].outputs[0].token_ids
    assert outputs[0].metrics is not None
    assert outputs[0].metrics.queued_ts > 0
    assert outputs[0].metrics.scheduled_ts > 0
    assert outputs[0].metrics.first_token_ts > 0
    assert outputs[0].metrics.first_token_ts >= outputs[0].metrics.scheduled_ts
