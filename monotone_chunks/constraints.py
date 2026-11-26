"""Human time-window constraints and monotone geometric feasibility."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .result import Result


@dataclass
class ConstraintSet:
    """Mutable active constraints with immutable revision history."""

    word_to_index: Mapping[str, int]
    records: dict[str, Result] = field(default_factory=dict)
    history: list[Result] = field(default_factory=list)

    @classmethod
    def from_materials(
        cls,
        word_ids: Iterable[str],
        records: Iterable[Mapping[str, Any]] | None = None,
    ) -> "ConstraintSet":
        word_to_index = {w: i for i, w in enumerate(word_ids)}
        obj = cls(word_to_index=word_to_index)
        if records is not None:
            obj.load_many(records)
        return obj

    def _word_index(self, rec: Mapping[str, Any]) -> int:
        if "word_index" in rec:
            wi = int(rec["word_index"])
            if not 0 <= wi < len(self.word_to_index):
                raise IndexError(f"word_index out of range: {wi}")
            return wi
        word = rec["word"]
        if word not in self.word_to_index:
            raise KeyError(f"unknown constrained word: {word}")
        return int(self.word_to_index[word])

    def load(self, rec: Mapping[str, Any]) -> Result:
        wi = self._word_index(rec)
        rid = str(rec.get("id", f"auto-{wi}-{len(self.records)}"))
        clean = Result(
            id=rid,
            revision=int(rec.get("revision", 1)),
            word=str(rec.get("word", next(w for w, i in self.word_to_index.items() if i == wi))),
            word_index=wi,
            lo=int(rec["lo"]),
            hi=int(rec["hi"]),
            outside_factor=float(rec.get("outside_factor", 1.0)),
        )
        if not 0 <= clean.lo <= clean.hi:
            raise ValueError("constraint requires 0 <= lo <= hi")
        if not 0.0 <= clean.outside_factor <= 1.0:
            raise ValueError("outside_factor must be in [0, 1]")
        old = self.records.get(rid)
        if old is not None:
            self.history.append(Result(old))
        self.records[rid] = clean
        self.history.append(Result(clean))
        return clean

    def load_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        for rec in records:
            self.load(rec)

    def revise(self, revision: Mapping[str, Any]) -> Result:
        rid = str(revision["record" if "record" in revision else "id"])
        if rid not in self.records:
            # A revision may itself contain enough fields to create the record.
            if "lo" not in revision or "hi" not in revision:
                raise KeyError(f"unknown constraint record: {rid}")
            payload = dict(revision)
            payload["id"] = rid
            return self.load(payload)
        old = self.records[rid]
        payload = dict(old)
        payload.update(dict(revision))
        payload["id"] = rid
        payload["revision"] = int(revision.get("revision", old.revision + 1))
        return self.load(payload)

    def active_records(self, word_index: int | None = None) -> list[Result]:
        vals = list(self.records.values())
        if word_index is not None:
            vals = [r for r in vals if r.word_index == word_index]
        return vals

    def factors_matrix(
        self,
        num_frames: int,
        num_words: int | None = None,
        *,
        stop: int | None = None,
    ) -> np.ndarray:
        n_words = num_words if num_words is not None else len(self.word_to_index)
        n = num_frames if stop is None else min(num_frames, int(stop))
        return build_constraint_matrix(
            n_words,
            n,
            self.active_records(),
            self.word_to_index,
        )

    def max_hi(self) -> int | None:
        if not self.records:
            return None
        return max(int(r.hi) for r in self.records.values())


def build_constraint_matrix(
    num_words: int,
    num_frames: int,
    constraints: Any | None = None,
    word_to_index: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Return ``C[word, frame]`` as a product of window factors."""
    c = np.ones((int(num_words), int(num_frames)), dtype=np.float64)
    if constraints is None:
        return c
    if isinstance(constraints, np.ndarray):
        arr = np.asarray(constraints, dtype=np.float64)
        if arr.shape != c.shape:
            # Allow callers to supply the full matrix while requesting a prefix.
            if arr.ndim == 2 and arr.shape[0] == num_words and arr.shape[1] >= num_frames:
                return arr[:, :num_frames].copy()
            raise ValueError(f"constraint matrix shape {arr.shape} != {c.shape}")
        return arr.copy()
    if isinstance(constraints, ConstraintSet):
        records = constraints.active_records()
        wmap = constraints.word_to_index
    elif isinstance(constraints, Mapping):
        # A single record or a mapping of word-index -> factor row.
        if "lo" in constraints and "hi" in constraints:
            records = [constraints]  # type: ignore[list-item]
            wmap = word_to_index or {}
        else:
            for wi, row in constraints.items():
                c[int(wi)] = np.asarray(row, dtype=np.float64)
            return c
    else:
        records = list(constraints)
        wmap = word_to_index or {}

    for rec in records:
        if isinstance(rec, Mapping):
            if "word_index" in rec:
                wi = int(rec["word_index"])
            elif wmap and "word" in rec:
                wi = int(wmap[str(rec["word"])])
            else:
                raise ValueError("constraint needs word_index or a word map")
            lo = int(rec["lo"])
            hi = int(rec["hi"])
            factor = float(rec.get("outside_factor", rec.get("weight", 1.0)))
            revision = rec.get("revision")
            active_revision = rec.get("active_revision")
            if active_revision is not None and revision is not None and int(revision) != int(active_revision):
                continue
        else:  # pragma: no cover - defensive
            wi = int(rec.word_index)
            lo, hi, factor = int(rec.lo), int(rec.hi), float(rec.outside_factor)
        if not 0 <= wi < num_words:
            continue
        # Build this record's own multiplicative factor vector.  Assigning 1
        # directly into c would erase factors from earlier overlapping records.
        record_factor = np.full(num_frames, factor, dtype=np.float64)
        left = max(0, lo)
        right = min(num_frames - 1, hi)
        if left <= right:
            record_factor[left : right + 1] = 1.0
        c[wi] *= record_factor
    return c


def one_legal_path(feasibility: Result) -> list[int] | None:
    """Backtrack one non-decreasing legal path from a feasibility DP result."""
    reachable = feasibility.get("reachable")
    if reachable is None or not bool(feasibility.feasible):
        return None
    w, t = reachable.shape
    if w == 0:
        return []
    j = int(np.flatnonzero(reachable[-1])[-1])
    path = [0] * w
    path[-1] = j
    for i in range(w - 1, 0, -1):
        candidates = np.flatnonzero(reachable[i - 1, : path[i] + 1])
        if candidates.size == 0:
            return None
        path[i - 1] = int(candidates[-1])
    return path


def geometric_feasibility(
    constraints: Any | None,
    num_frames: int,
    *,
    valid: np.ndarray | None = None,
    start_boundary: int = 0,
    frame_mask: np.ndarray | None = None,
    num_words: int | None = None,
) -> Result:
    """Boolean monotone DP; one word having frames is not enough.

    ``frame_mask`` can represent currently confirmed/received frames.  ``valid``
    represents non-padding acoustic positions.  The returned reachable rows let
    callers distinguish a temporarily uncovered prefix from a permanent conflict.
    """
    if isinstance(constraints, np.ndarray):
        c = constraints
        n_words = c.shape[0]
        if num_frames is None:
            num_frames = c.shape[1]
        elif c.shape[1] > num_frames:
            c = c[:, :num_frames]
    else:
        n_words = num_words
        if n_words is None and isinstance(constraints, ConstraintSet):
            n_words = len(constraints.word_to_index)
        if n_words is None:
            raise ValueError("num_words is required for record constraints")
        c = build_constraint_matrix(n_words, num_frames, constraints)

    n = int(num_frames)
    if c.shape[1] != n:
        c = c[:, :n]
    acoustic = np.ones(n, dtype=bool) if valid is None else np.asarray(valid[:n], dtype=bool)
    if frame_mask is not None:
        acoustic &= np.asarray(frame_mask[:n], dtype=bool)
    hard_allowed = acoustic[None, :] & (c > 0.0)

    reachable_words = np.zeros((n_words, n), dtype=bool)
    if n == 0 or not 0 <= start_boundary < n:
        return Result(
            feasible=False,
            reason="no_start_frame",
            reachable=reachable_words,
            final=np.zeros(n, dtype=bool),
            allowed_path_count=0,
        )

    previous = np.zeros(n, dtype=bool)
    previous[start_boundary] = True
    count_previous = np.zeros(n, dtype=np.float64)
    count_previous[start_boundary] = 1.0
    path_count_saturated = False

    for i in range(n_words):
        cumulative = np.maximum.accumulate(previous)
        reach = hard_allowed[i] & cumulative
        reachable_words[i] = reach

        # Saturating DP count (not path enumeration).  Float is only used to
        # avoid giant integers in capacity runs; exact small counts are common.
        cum_count = np.cumsum(count_previous)
        count_row = np.where(hard_allowed[i], cum_count, 0.0)
        if np.any(count_row > 1e100):
            path_count_saturated = True
            count_row = np.minimum(count_row, 1e100)
        count_previous = count_row
        previous = reach

    final = reachable_words[-1] if n_words else previous
    feasible = bool(final.any())
    if not feasible:
        # Identify whether any individual word has a legal frame.  This does
        # not establish sentence feasibility, but is useful in diagnostics.
        individual = [bool(hard_allowed[i].any()) for i in range(n_words)]
        if all(individual):
            reason = "joint_monotone_constraints_infeasible"
        elif not acoustic.any():
            reason = "no_valid_acoustic_frame"
        else:
            reason = "word_constraint_has_no_allowed_frame"
    else:
        reason = "feasible"
    return Result(
        feasible=feasible,
        reason=reason,
        reachable=reachable_words,
        final=final,
        allowed_path_count=float(count_previous[final].sum()) if n_words else 1.0,
        path_count_saturated=path_count_saturated,
        hard_allowed=hard_allowed,
    )
