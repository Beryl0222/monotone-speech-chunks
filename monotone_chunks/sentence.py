"""Input container and energy composition from the data contract."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constraints import ConstraintSet
from .result import Result


@dataclass
class SentenceInputs:
    word_ids: list[str]
    Ks: np.ndarray
    Kc: np.ndarray
    V: np.ndarray
    Q: np.ndarray
    b_select: np.ndarray | float
    chunk_width: int
    valid: np.ndarray
    utterance_id: str = "utterance"
    sample_weight: float = 1.0
    context_upstream: np.ndarray | None = None
    tier: dict[str, Any] | None = None
    constraints: ConstraintSet | None = None
    scale: float | None = None

    @classmethod
    def from_materials(cls, data: Mapping[str, Any], *, load_constraints: bool = True) -> "SentenceInputs":
        word_ids = list(data["word_ids"])
        w = len(word_ids)
        obj = cls(
            word_ids=word_ids,
            Ks=np.asarray(data["Ks"], dtype=np.float64),
            Kc=np.asarray(data["Kc"], dtype=np.float64),
            V=np.asarray(data["V"], dtype=np.float64),
            Q=np.asarray(data["Q"], dtype=np.float64),
            b_select=np.asarray(data.get("b_select", np.zeros(w)), dtype=np.float64),
            chunk_width=int(data["chunk_width"]),
            valid=np.asarray(data.get("valid", np.ones(len(data["V"]))), dtype=bool),
            utterance_id=str(data.get("utterance_id", "utterance")),
            sample_weight=float(data.get("sample_weight", 1.0)),
            context_upstream=(
                None
                if "context_upstream" not in data
                else np.asarray(data["context_upstream"], dtype=np.float64)
            ),
            tier=dict(data["online_tier"]) if data.get("online_tier") is not None else None,
            scale=None if "attention_scale" not in data else float(data["attention_scale"]),
        )
        if load_constraints and data.get("word_constraints") is not None:
            obj.constraints = ConstraintSet.from_materials(word_ids, data["word_constraints"])
        return obj

    @classmethod
    def from_json(cls, path: str | Path) -> "SentenceInputs":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_materials(data)

    @classmethod
    def from_capacity_dir(cls, directory: str | Path) -> "SentenceInputs":
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "arrays.npz")
        words = list(meta.get("words", []))
        q = arrays["Q"]
        if not words:
            words = [f"w{i}" for i in range(q.shape[0])]
        valid = arrays["valid"] if "valid" in arrays else np.ones(arrays["V"].shape[0], dtype=bool)
        return cls(
            word_ids=words,
            Ks=arrays["Ks"],
            Kc=arrays["Kc"],
            V=arrays["V"],
            Q=q,
            b_select=arrays["b_select"] if "b_select" in arrays else np.zeros(q.shape[0]),
            chunk_width=int(meta["chunk_width"]),
            valid=valid.astype(bool),
            utterance_id=str(meta.get("utterance_id", "capacity")),
            sample_weight=float(meta.get("sample_weight", 1.0)),
            context_upstream=arrays["context_upstream"] if "context_upstream" in arrays else None,
            tier=meta.get("online_tier"),
        )

    @property
    def num_frames(self) -> int:
        return int(self.Ks.shape[0])

    @property
    def num_words(self) -> int:
        return int(self.Q.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.Q.shape[1])

    @property
    def value_dim(self) -> int:
        return int(self.V.shape[1])

    @property
    def attention_scale(self) -> float:
        return self.scale if self.scale is not None else 1.0 / np.sqrt(self.head_dim)

    def bias_for(self, word_index: int) -> float:
        b = self.b_select
        if np.isscalar(b):
            return float(b)
        b_arr = np.asarray(b, dtype=np.float64)
        if b_arr.ndim == 0:
            return float(b_arr)
        if b_arr.shape[0] == self.num_words:
            return float(b_arr[word_index])
        if b_arr.shape[0] == self.num_frames:
            return 0.0  # frame bias would be added by caller-provided e; not here
        return float(b_arr.reshape(-1)[0])

    def selection_energy_row(self, word_index: int, stop: int | None = None) -> np.ndarray:
        n = self.num_frames if stop is None else min(self.num_frames, int(stop))
        row = self.Q[word_index] @ self.Ks[:n].T
        return row + self.bias_for(word_index)

    def chunk_energy_row(self, word_index: int, stop: int | None = None) -> np.ndarray:
        n = self.num_frames if stop is None else min(self.num_frames, int(stop))
        return (self.Q[word_index] @ self.Kc[:n].T) * self.attention_scale

    def selection_energies(self, stop: int | None = None) -> np.ndarray:
        n = self.num_frames if stop is None else min(self.num_frames, int(stop))
        bias = np.asarray(self.b_select, dtype=np.float64)
        if bias.ndim == 0:
            bias2 = float(bias)
        elif bias.shape[0] == self.num_words:
            bias2 = bias[:, None]
        else:
            bias2 = 0.0
        return self.Q @ self.Ks[:n].T + bias2

    def chunk_energies(self, stop: int | None = None) -> np.ndarray:
        n = self.num_frames if stop is None else min(self.num_frames, int(stop))
        return (self.Q @ self.Kc[:n].T) * self.attention_scale

    def word_index(self, word: str | int) -> int:
        if isinstance(word, int):
            return word
        return self.word_ids.index(str(word))

    def last_valid_frame(self, stop: int | None = None) -> int | None:
        n = self.num_frames if stop is None else min(self.num_frames, int(stop))
        positions = np.flatnonzero(np.asarray(self.valid[:n], dtype=bool))
        return int(positions[-1]) if positions.size else None

    def mask(self, arrived_stop: int | None = None, confirmed_stop: int | None = None) -> np.ndarray:
        n = self.num_frames if arrived_stop is None else min(self.num_frames, int(arrived_stop))
        mask = np.asarray(self.valid[:n], dtype=bool).copy()
        if confirmed_stop is not None:
            conf = np.zeros(n, dtype=bool)
            conf[: min(n, int(confirmed_stop))] = True
            mask &= conf
        return mask
