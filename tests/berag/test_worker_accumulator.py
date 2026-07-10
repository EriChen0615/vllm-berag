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
        mixture_row_id=0,
        scheduled_req_ids=["parent:berag:4"],
        scheduled_branch_ids=[4],
        prior_req_ids=[],
        prior_branch_ids=[],
        prior_token_indices=[],
        evidence_branch_ids=[4],
        evidence_row_ids=[5],
        mix_req_ids=[f"parent:berag:{branch_id}" for branch_id in range(5)],
        mix_branch_ids=[0, 1, 2, 3, 4],
        mix_row_ids=[1, 2, 3, 4, 5],
        log_posterior=[0.0, -1.0, -2.0, -3.0, -4.0],
        is_final_shard=True,
        sample_on_completion=True,
    )


def make_direct_mix_shard(*, with_worker_priors: bool = False) -> ScheduledBeragShard:
    return ScheduledBeragShard(
        group_id="parent",
        step_id=0,
        mixture_row_id=-1,
        scheduled_req_ids=["parent:berag:0", "parent:berag:1"],
        scheduled_branch_ids=[0, 1],
        prior_req_ids=["parent:berag:0", "parent:berag:1"]
        if with_worker_priors
        else [],
        prior_branch_ids=[0, 1] if with_worker_priors else [],
        prior_token_indices=[0, 0] if with_worker_priors else [],
        evidence_branch_ids=[0, 1],
        evidence_row_ids=[],
        mix_req_ids=["parent:berag:0", "parent:berag:1"],
        mix_branch_ids=[0, 1],
        mix_row_ids=[],
        log_posterior=[] if with_worker_priors else [0.0, -1.0],
        is_final_shard=True,
        direct_mix=True,
        sample_on_completion=True,
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


def test_v1_worker_direct_mix_samples_without_accumulator_on_cpu():
    logits = torch.tensor([[0.0, 4.0, -2.0], [4.0, 0.0, -2.0]])
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_berag_shards = [make_direct_mix_shard()]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["parent:berag:0", "parent:berag:1"],
        ),
        berag_accumulator=None,
        device=torch.device("cpu"),
        _berag_debug_shard=lambda *args, **kwargs: None,
        _sample_berag_mixture=lambda mixture, req_id: 1,
    )

    outputs = V1GPUModelRunner._process_berag_shards(
        runner,
        scheduler_output,
        logits,
        hidden_states=None,
    )

    expected = logits.log_softmax(dim=-1).to(torch.bfloat16)[:, 1].float()
    assert outputs[0].completed_branch_ids == [0, 1]
    assert outputs[0].sampled_token_id == 1
    assert outputs[0].sampled_token_logprobs == pytest.approx(
        {0: float(expected[0]), 1: float(expected[1])}
    )


def test_v1_worker_direct_mix_computes_priors_and_samples_on_cpu():
    logits = torch.tensor([[0.0, 4.0, -2.0], [4.0, 0.0, -2.0]])
    hidden_states = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_berag_shards = [
        make_direct_mix_shard(with_worker_priors=True)
    ]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["parent:berag:0", "parent:berag:1"],
            num_computed_tokens_cpu=[0, 0],
        ),
        query_start_loc=SimpleNamespace(np=[0, 1]),
        berag_child_by_req_id={
            "parent:berag:0": SimpleNamespace(branch_id=0),
            "parent:berag:1": SimpleNamespace(branch_id=1),
        },
        berag_accumulator=None,
        device=torch.device("cpu"),
        _berag_prior_score=lambda hidden: float(hidden[0].float().item()),
        _berag_debug_shard=lambda *args, **kwargs: None,
        _sample_berag_mixture=lambda mixture, req_id: 1,
    )

    outputs = V1GPUModelRunner._process_berag_shards(
        runner,
        scheduler_output,
        logits,
        hidden_states=hidden_states,
    )

    assert outputs[0].prior_scores == {0: 2.0, 1: 0.0}
    assert outputs[0].sampled_token_id == 1


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


def test_v2_worker_direct_mix_samples_without_accumulator_on_cpu():
    logits = torch.tensor([[0.0, 4.0, -2.0], [4.0, 0.0, -2.0]])
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_berag_shards = [make_direct_mix_shard()]
    input_batch = SimpleNamespace(
        req_ids=["parent:berag:0", "parent:berag:1"],
        logits_indices=torch.tensor([0, 1]),
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda hidden_states: logits),
        berag_accumulator=None,
        device=torch.device("cpu"),
        _berag_debug_shard=lambda *args, **kwargs: None,
        _sample_berag_mixture=lambda mixture, req_id: 1,
    )

    outputs = V2GPUModelRunner._process_berag_shards(
        runner,
        scheduler_output,
        input_batch,
        hidden_states=torch.zeros((2, 2)),
    )

    expected = logits.log_softmax(dim=-1).to(torch.bfloat16)[:, 1].float()
    assert outputs[0].completed_branch_ids == [0, 1]
    assert outputs[0].sampled_token_id == 1
    assert outputs[0].sampled_token_logprobs == pytest.approx(
        {0: float(expected[0]), 1: float(expected[1])}
    )


def test_v2_worker_direct_mix_computes_priors_and_samples_on_cpu():
    logits = torch.tensor([[0.0, 4.0, -2.0], [4.0, 0.0, -2.0]])
    hidden_states = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_berag_shards = [
        make_direct_mix_shard(with_worker_priors=True)
    ]
    input_batch = SimpleNamespace(
        req_ids=["parent:berag:0", "parent:berag:1"],
        logits_indices=torch.tensor([0, 1]),
        query_start_loc_np=[0, 1],
        num_computed_tokens_np=[0, 0],
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda sample_hidden_states: logits),
        berag_child_by_req_id={
            "parent:berag:0": SimpleNamespace(branch_id=0),
            "parent:berag:1": SimpleNamespace(branch_id=1),
        },
        berag_accumulator=None,
        device=torch.device("cpu"),
        _berag_prior_score=lambda hidden: float(hidden[0].float().item()),
        _berag_debug_shard=lambda *args, **kwargs: None,
        _sample_berag_mixture=lambda mixture, req_id: 1,
    )

    outputs = V2GPUModelRunner._process_berag_shards(
        runner,
        scheduler_output,
        input_batch,
        hidden_states=hidden_states,
    )

    assert outputs[0].prior_scores == {0: 2.0, 1: 0.0}
    assert outputs[0].sampled_token_id == 1
