# SPDX-License-Identifier: Apache-2.0
"""Warm up vLLM with ordinary generation requests.

This intentionally does not use BERAG. It exercises the same model runner and
KV-cache slot-mapping path with a small batch so Triton/CUDA caches are populated
before a BERAG smoke run.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".cache"
TMP_ROOT = REPO_ROOT / ".tmp"


def set_cache_env() -> None:
    env_defaults = {
        "TMPDIR": TMP_ROOT,
        "TEMP": TMP_ROOT,
        "TMP": TMP_ROOT,
        "XDG_CACHE_HOME": CACHE_ROOT,
        "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
        "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
        "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT / "torchinductor",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, str(value))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def make_prompts(num_prompts: int, repeat: int) -> list[str]:
    documents = [
        "Document: The daytime sky is usually blue.",
        "Document: Grass is often green.",
        "Document: Snow is often white.",
        "Document: Lemons are usually yellow.",
        "Document: Fire trucks are often red.",
        "Document: Deep ocean water can look dark blue.",
    ]
    prompts = []
    for index in range(num_prompts):
        document = " ".join([documents[index % len(documents)]] * repeat)
        prompts.append(f"Question: What color is the sky?\n{document}\nAnswer:")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("BERAG_E2E_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    set_cache_env()

    from vllm import LLM, SamplingParams

    args = parse_args()
    prompts = make_prompts(args.num_prompts, args.prompt_repeat)

    print("[warmup] cache dirs:")
    for key in (
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
    ):
        print(f"[warmup]   {key}={os.environ[key]}")

    print(f"[warmup] loading model={args.model}")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    for run in range(args.runs):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        print(
            f"[warmup] run={run + 1}/{args.runs} "
            f"elapsed={elapsed:.2f}s output_token_counts={token_counts}"
        )
        for index, output in enumerate(outputs):
            completion = output.outputs[0]
            print(
                f"[warmup]   prompt={index} "
                f"token_ids={list(completion.token_ids)} "
                f"text={completion.text!r}"
            )

    print("[warmup] done")


if __name__ == "__main__":
    main()
