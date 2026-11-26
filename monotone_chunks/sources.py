"""Small replayable acoustic source over in-memory arrays or NPZ files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .workspace import ReplayBudget


class ReplayableAcousticSource:
    """Read confirmed K/Kc/V ranges without storing decoder score histories.

    NPZ inputs are memory-mapped where possible.  Every block-group read can
    consume a caller-supplied :class:`ReplayBudget`, making the "at most 64
    acoustic block passes" replay limit explicit.
    """

    def __init__(
        self,
        Ks: np.ndarray | Any,
        Kc: np.ndarray | Any,
        V: np.ndarray | Any,
        valid: np.ndarray | None = None,
        *,
        block_size: int = 64,
        budget: ReplayBudget | None = None,
    ) -> None:
        self.Ks = Ks
        self.Kc = Kc
        self.V = V
        self.valid = np.ones(V.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self.block_size = int(block_size)
        self.budget = budget

    @classmethod
    def from_npz(cls, path: str | Path, *, block_size: int = 64, mmap: bool = True) -> "ReplayableAcousticSource":
        npz = np.load(path, mmap_mode="r" if mmap else None)
        return cls(
            npz["Ks"],
            npz["Kc"],
            npz["V"],
            npz["valid"] if "valid" in npz.files else None,
            block_size=block_size,
        )

    def read(self, start: int = 0, stop: int | None = None, *, count_pass: bool = False):
        if stop is None:
            stop = self.V.shape[0]
        if count_pass and self.budget is not None:
            blocks = max(1, int(np.ceil((stop - start) / self.block_size)))
            self.budget.consume(blocks)
        return (
            np.asarray(self.Ks[start:stop]),
            np.asarray(self.Kc[start:stop]),
            np.asarray(self.V[start:stop]),
            np.asarray(self.valid[start:stop]),
        )

    def block_count(self, stop: int | None = None) -> int:
        n = self.V.shape[0] if stop is None else int(stop)
        return int(np.ceil(n / self.block_size))
