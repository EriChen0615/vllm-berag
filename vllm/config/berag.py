# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from pydantic import Field

from vllm.config.utils import config


@config
class BeragConfig:
    """Engine-level BERAG configuration."""

    num_accumulator_rows: int = Field(default=400, ge=2)
    prior_module_cls: str | None = None
    prior_module_weights_path: str | None = None
    prior_module_kwargs: dict[str, Any] = Field(default_factory=dict)
    default_prior_token_offset: int = -4

    @property
    def enabled(self) -> bool:
        return (
            self.prior_module_cls is not None
            or self.prior_module_weights_path is not None
        )
