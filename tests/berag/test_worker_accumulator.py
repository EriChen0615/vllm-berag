# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.model_runner import BeragAccumulator


def test_berag_accumulator_tracks_live_rows_on_cpu():
    accumulator = BeragAccumulator(
        num_rows=3,
        vocab_size=5,
        device=torch.device("cpu"),
    )

    assert accumulator.workspace.shape == (3, 5)
    assert accumulator.workspace.dtype == torch.bfloat16

    accumulator.mark_live(0)
    accumulator.mark_live(2)
    telemetry = accumulator.telemetry()
    assert telemetry.total_rows == 3
    assert telemetry.live_rows == 2
    assert telemetry.free_rows == 1

    accumulator.release([0])
    telemetry = accumulator.telemetry()
    assert telemetry.live_rows == 1
    assert telemetry.free_rows == 2


def test_berag_logprob_mixture_matches_logsumexp():
    workspace = torch.empty((3, 4), dtype=torch.bfloat16)
    workspace[0].copy_(torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    workspace[1].copy_(torch.tensor([-2.0, -1.0, -4.0, -3.0]))
    log_weights = torch.tensor([[0.0], [-1.0]], dtype=torch.bfloat16)

    mixture = torch.logsumexp(workspace[:2] + log_weights, dim=0)

    expected = torch.logsumexp(
        torch.tensor(
            [
                [-1.0, -2.0, -3.0, -4.0],
                [-3.0, -2.0, -5.0, -4.0],
            ]
        ),
        dim=0,
    )
    torch.testing.assert_close(mixture.float(), expected, rtol=0.01, atol=0.01)
