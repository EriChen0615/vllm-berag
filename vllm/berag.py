# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import msgspec


@dataclass
class BeragParams:
    """Per-request BERAG parameters."""

    pruning_top_p: float = 1.0
    prior_token_indices: list[int] | None = None


class BeragChildMetadata(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    group_id: str
    parent_request_id: str
    branch_id: int
    num_branches: int
    parent_prompt_len: int
    prior_token_index: int
    pruning_top_p: float
    debug: bool = False
