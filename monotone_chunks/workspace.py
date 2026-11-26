"""Contract limits and a small replay budget helper.

The numerical core uses O(T) recurrence state per decoded word and retains the
returned W x T boundary distribution.  It never enumerates paths.  This module
records the public capacity envelope so callers can fail early rather than
accidentally allocating an exponential or oversized workspace.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .result import Result

LIMITS = Result(
    max_frames=4096,
    max_words=128,
    max_dim=16,
    max_chunk_width=64,
    block_size=64,
    max_workspace_bytes=4 * 1024 * 1024,
    max_replay_passes=64,
)


class LimitExceeded(ValueError):
    pass


def validate_shapes(
    num_frames: int,
    num_words: int,
    head_dim: int,
    chunk_width: int,
    *,
    strict: bool = False,
) -> Result:
    problems = []
    if num_frames < 0 or num_words < 0 or head_dim < 1 or chunk_width < 1:
        raise ValueError("shape dimensions must be non-negative and widths positive")
    if num_frames > LIMITS.max_frames:
        problems.append(f"frames {num_frames} > {LIMITS.max_frames}")
    if num_words > LIMITS.max_words:
        problems.append(f"words {num_words} > {LIMITS.max_words}")
    if head_dim > LIMITS.max_dim:
        problems.append(f"dim {head_dim} > {LIMITS.max_dim}")
    if chunk_width > LIMITS.max_chunk_width:
        problems.append(f"chunk_width {chunk_width} > {LIMITS.max_chunk_width}")
    if strict and problems:
        raise LimitExceeded("; ".join(problems))
    alpha_bytes = int(num_words) * int(num_frames) * np.dtype(np.float64).itemsize
    return Result(ok=not problems, problems=problems, alpha_bytes=alpha_bytes, limits=LIMITS)


@dataclass
class ReplayBudget:
    """Count acoustic-block passes for one replay request."""

    max_passes: int = LIMITS.max_replay_passes
    passes: int = 0

    def consume(self, count: int = 1) -> None:
        if self.passes + count > self.max_passes:
            raise LimitExceeded(
                f"replay passes {self.passes + count} exceed max {self.max_passes}"
            )
        self.passes += int(count)

    def remaining(self) -> int:
        return self.max_passes - self.passes
