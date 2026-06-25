# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.berag.prior_fixtures import TinyPrior
from vllm.config.berag import BeragConfig
from vllm.v1.core.sched.output import ScheduledBeragShard, SchedulerOutput
from vllm.v1.worker.gpu.model_runner import BeragAccumulator as V2BeragAccumulator
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
from vllm.v1.worker.gpu_model_runner import BeragAccumulator as V1BeragAccumulator
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1GPUModelRunner


@pytest.mark.parametrize("accumulator_cls", [V1BeragAccumulator, V2BeragAccumulator])
def test_berag_accumulator_tracks_live_rows_on_cpu(accumulator_cls):
    accumulator = accumulator_cls(
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


@pytest.mark.parametrize("runner_cls", [V1GPUModelRunner, V2GPUModelRunner])
def test_berag_prior_module_loads_class_path_weights_on_cpu(tmp_path, runner_cls):
    prior = TinyPrior(hidden_size=4)
    weights_path = tmp_path / "prior.pt"
    torch.save(prior.state_dict(), weights_path)
    runner = SimpleNamespace(
        berag_config=BeragConfig(
            prior_module_cls="tests.berag.prior_fixtures.TinyPrior",
            prior_module_weights_path=str(weights_path),
            prior_module_kwargs={"hidden_size": 4},
        ),
        device=torch.device("cpu"),
        berag_prior_module=None,
    )

    runner_cls._load_berag_prior_module(runner)

    assert isinstance(runner.berag_prior_module, TinyPrior)
    assert not runner.berag_prior_module.training
    assert next(runner.berag_prior_module.parameters()).dtype == torch.bfloat16


@pytest.mark.parametrize("runner_cls", [V1GPUModelRunner, V2GPUModelRunner])
def test_uniform_berag_prior_skips_module_and_returns_zero_on_cpu(runner_cls):
    runner = SimpleNamespace(
        berag_config=BeragConfig(prior_mode="uniform"),
        device=torch.device("cpu"),
        berag_prior_module=None,
    )

    runner_cls._load_berag_prior_module(runner)

    assert runner.berag_prior_module is None
    assert runner_cls._berag_prior_score(runner, torch.ones(4)) == 0.0


def make_final_subset_shard() -> ScheduledBeragShard:
    return ScheduledBeragShard(
        group_id="parent",
        step_id=1,
        req_ids=["parent:berag:4"],
        branch_ids=[0, 1, 2, 3, 4],
        mixture_row_id=0,
        branch_row_ids=[1, 2, 3, 4, 5],
        log_posterior=[0.0, -1.0, -2.0, -3.0, -4.0],
        is_final_shard=True,
        sample_on_completion=True,
        scheduled_branch_ids=[4],
    )


def seed_branch_rows(accumulator) -> None:
    accumulator.workspace[1].copy_(torch.tensor([-0.1, -3.0, -4.0]))
    accumulator.workspace[2].copy_(torch.tensor([-3.0, -0.2, -4.0]))
    accumulator.workspace[3].copy_(torch.tensor([-4.0, -3.0, -0.3]))
    accumulator.workspace[4].copy_(torch.tensor([-0.4, -4.0, -3.0]))


def assert_final_subset_worker_output(accumulator, output) -> None:
    row5 = torch.tensor([[0.0, 4.0, -2.0]]).log_softmax(dim=-1).to(
        torch.bfloat16
    )[0]
    expected_rows = torch.vstack([accumulator.workspace[row] for row in range(1, 6)])
    expected_weights = torch.tensor(
        [[0.0], [-1.0], [-2.0], [-3.0], [-4.0]],
        dtype=torch.bfloat16,
    )
    expected_mixture = torch.logsumexp(expected_rows + expected_weights, dim=0)

    assert output.completed_branch_ids == [4]
    assert output.sampled_token_id == 1
    assert set(output.sampled_token_logprobs) == {0, 1, 2, 3, 4}
    torch.testing.assert_close(accumulator.workspace[5].float(), row5.float())
    torch.testing.assert_close(
        accumulator.workspace[0].float(),
        expected_mixture.float(),
        rtol=0.01,
        atol=0.01,
    )


def test_v1_worker_processes_final_shard_with_scheduled_branch_subset_on_cpu():
    accumulator = V1BeragAccumulator(6, 3, torch.device("cpu"))
    seed_branch_rows(accumulator)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_berag_shards = [make_final_subset_shard()]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["parent:berag:4"]),
        berag_accumulator=accumulator,
        device=torch.device("cpu"),
        _berag_debug_shard=lambda *args, **kwargs: None,
        _sample_berag_mixture=lambda mixture, req_id: 1,
    )

    outputs = V1GPUModelRunner._process_berag_shards(
        runner,
        scheduler_output,
        torch.tensor([[0.0, 4.0, -2.0]]),
        hidden_states=None,
    )

    assert_final_subset_worker_output(accumulator, outputs[0])


def test_v2_worker_processes_final_shard_with_scheduled_branch_subset_on_cpu():
    accumulator = V2BeragAccumulator(6, 3, torch.device("cpu"))
    seed_branch_rows(accumulator)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_berag_shards = [make_final_subset_shard()]
    input_batch = SimpleNamespace(
        req_ids=["parent:berag:4"],
        logits_indices=torch.tensor([0]),
    )
    model = SimpleNamespace(
        compute_logits=lambda hidden_states: torch.tensor([[0.0, 4.0, -2.0]])
    )
    runner = SimpleNamespace(
        model=model,
        berag_accumulator=accumulator,
        device=torch.device("cpu"),
        _berag_debug_shard=lambda *args, **kwargs: None,
        _sample_berag_mixture=lambda mixture, req_id: 1,
    )

    outputs = V2GPUModelRunner._process_berag_shards(
        runner,
        scheduler_output,
        input_batch,
        hidden_states=torch.zeros((1, 2)),
    )

    assert_final_subset_worker_output(accumulator, outputs[0])
