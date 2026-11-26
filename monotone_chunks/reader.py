"""Stateful streaming read/submission state machine."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .boundary import boundary_distribution, sigmoid
from .constraints import ConstraintSet, geometric_feasibility
from .result import Result
from .sentence import SentenceInputs
from .windows import window_range, window_attention


class Proposal(Result):
    """A read proposal.  Attribute access is provided by :class:`Result`."""


class StreamingReader:
    """Continuously fed audio, threshold reads, previews, commits, withdrawals.

    The formal cursor is always the last *committed* boundary.  Provisional
    boundaries form a disposable chain and are never used after withdrawal.
    """

    def __init__(
        self,
        sentence: SentenceInputs | None = None,
        *,
        word_ids: list[str] | None = None,
        chunk_width: int | None = None,
        tier: Mapping[str, Any] | None = None,
        tau: float | None = None,
        max_lookahead: int | None = None,
        constraints: ConstraintSet | list[Mapping[str, Any]] | None = None,
        lookahead_reference: str = "acquired_end",
        e: np.ndarray | None = None,
        selection_energy: np.ndarray | None = None,
        selection_scores: np.ndarray | None = None,
        selection_probabilities: np.ndarray | None = None,
        p: np.ndarray | None = None,
        u: np.ndarray | None = None,
        chunk_energy: np.ndarray | None = None,
        chunk_scores: np.ndarray | None = None,
        V: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        num_frames: int | None = None,
        Q: np.ndarray | None = None,
        Ks: np.ndarray | None = None,
        Kc: np.ndarray | None = None,
        b_select: np.ndarray | float = 0.0,
        utterance_id: str = "direct",
        sample_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if chunk_width is None:
            chunk_width = kwargs.pop("chunk_size", kwargs.pop("width", None))
        if tau is None:
            tau = kwargs.pop("threshold", kwargs.pop("tau_threshold", None))
        if max_lookahead is None:
            max_lookahead = kwargs.pop(
                "lookahead",
                kwargs.pop("L", kwargs.pop("max_lookahead_frames", kwargs.pop("lookahead_frames", None))),
            )
        if "latency_tier" in kwargs and tier is None:
            tier = kwargs.pop("latency_tier")
        if isinstance(sentence, Mapping) and "Ks" in sentence:
            materials = dict(sentence)
            tier = tier or materials.get("online_tier")
            sentence = SentenceInputs.from_materials(materials)
        elif isinstance(sentence, Mapping):
            config = dict(sentence)
            tier = tier or config.get("tier") or config
            if tau is None:
                tau = config.get("tau")
            if max_lookahead is None:
                max_lookahead = config.get("max_lookahead")
            if "lookahead_reference" in config:
                lookahead_reference = config["lookahead_reference"]
            if word_ids is None and "word_ids" in config:
                word_ids = config["word_ids"]
            if chunk_width is None and "chunk_width" in config:
                chunk_width = config["chunk_width"]
            sentence = None
        if sentence is None and Q is not None and Ks is not None and Kc is not None and V is not None:
            q_arr = np.asarray(Q)
            w = q_arr.shape[0]
            if word_ids is None:
                word_ids = [f"w{i}" for i in range(w)]
            if chunk_width is None:
                chunk_width = 1
            t = np.asarray(Ks).shape[0]
            sentence = SentenceInputs(
                word_ids=list(word_ids),
                Ks=np.asarray(Ks, dtype=np.float64),
                Kc=np.asarray(Kc, dtype=np.float64),
                V=np.asarray(V, dtype=np.float64),
                Q=q_arr.astype(np.float64),
                b_select=b_select,
                chunk_width=int(chunk_width),
                valid=np.ones(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
                utterance_id=utterance_id,
                sample_weight=sample_weight,
            )
        if sentence is None:
            direct_e_init = e if e is not None else (selection_energy if selection_energy is not None else selection_scores)
            direct_p_init = p if p is not None else selection_probabilities
            if word_ids is None:
                w_shape = None
                for arr in (direct_e_init, direct_p_init, u, chunk_energy, chunk_scores):
                    if arr is not None:
                        w_shape = np.asarray(arr).shape[0]
                        break
                if w_shape is None:
                    raise ValueError("provide a SentenceInputs, word_ids, or score arrays")
                word_ids = [f"w{i}" for i in range(int(w_shape))]
            if chunk_width is None:
                chunk_width = 1
            self.sentence = None
            self.word_ids = list(word_ids)
            self.chunk_width = int(chunk_width)
            self.constraints = (
                constraints
                if isinstance(constraints, ConstraintSet)
                else ConstraintSet.from_materials(self.word_ids, constraints)
            )
        else:
            self.sentence = sentence
            self.word_ids = list(sentence.word_ids)
            self.chunk_width = int(chunk_width or sentence.chunk_width)
            if constraints is None:
                self.constraints = sentence.constraints or ConstraintSet.from_materials(self.word_ids)
            elif isinstance(constraints, ConstraintSet):
                self.constraints = constraints
            else:
                self.constraints = ConstraintSet.from_materials(self.word_ids, constraints)

        self.word_to_index = {w: i for i, w in enumerate(self.word_ids)}
        self.direct_e = e if e is not None else (selection_energy if selection_energy is not None else selection_scores)
        self.direct_p = p if p is not None else selection_probabilities
        self.direct_u = u if u is not None else (chunk_energy if chunk_energy is not None else chunk_scores)
        self.direct_V = V
        if valid is None and self.sentence is not None:
            valid = self.sentence.valid
        self.direct_valid = None if valid is None else np.asarray(valid, dtype=bool)
        if num_frames is None:
            for arr in (self.direct_e, self.direct_p, self.direct_u, self.direct_V):
                if arr is not None:
                    num_frames = np.asarray(arr).shape[1] if np.asarray(arr).ndim == 2 else np.asarray(arr).shape[0]
                    break
        self.direct_num_frames = num_frames
        tier_obj = dict(tier or (self.sentence.tier if self.sentence else {}) or {})
        self.tier = tier_obj
        self.tau = float(tau if tau is not None else tier_obj.get("tau", 0.5))
        self.max_lookahead = int(
            max_lookahead if max_lookahead is not None else tier_obj.get("max_lookahead", 0)
        )
        # Default latency reference is the acquired-audio end: n-j is the
        # right/look-ahead distance. "confirmed_frontier" is also available for
        # tiers that cap previews beyond the immutable confirmation line.
        self.lookahead_reference = lookahead_reference

        self.arrived_stop = 0
        self.confirmed_stop = 0
        self.sealed = False
        self.last_valid_frame: int | None = None
        self.audio_events: list[Result] = []

        self.committed: dict[int, Result] = {}
        self.committed_order: list[int] = []
        self.provisional: dict[int, Proposal] = {}
        self.subtitles: list[Result] = []
        self.interpretation_versions: list[Result] = []
        self.operation_log: list[Result] = []

    @classmethod
    def from_materials(cls, materials: Mapping[str, Any], **kwargs: Any) -> "StreamingReader":
        sentence = SentenceInputs.from_materials(materials)
        return cls(sentence, **kwargs)

    # ------------------------------------------------------------------ audio
    def audio_arrive(
        self,
        start: int | None = None,
        stop: int | None = None,
        confirmed_stop: int | None = None,
        chunk: Mapping[str, Any] | None = None,
        **payload: Any,
    ) -> Result:
        if isinstance(start, Mapping):
            chunk = start
            start = None
        if chunk is not None:
            start = chunk.get("start", start)
            stop = chunk.get("stop", stop)
            confirmed_stop = chunk.get("confirmed_stop", confirmed_stop)
        if start is not None and int(start) != self.arrived_stop:
            # The supplied event stream is contiguous.  Be permissive with
            # overlapping/reordered callers but never move a frontier backward.
            if int(start) > self.arrived_stop:
                self.arrived_stop = int(start)
        if stop is not None:
            if int(stop) < self.arrived_stop and int(start or 0) <= self.arrived_stop:
                # Overlap is allowed; retain the larger end.
                pass
            self.arrived_stop = max(self.arrived_stop, int(stop))
        if confirmed_stop is not None:
            self.confirmed_stop = max(self.confirmed_stop, int(confirmed_stop))
        event = Result(
            op="audio_arrive",
            start=int(start if start is not None else 0),
            stop=self.arrived_stop if stop is None else int(stop),
            arrived_stop=self.arrived_stop,
            confirmed_stop=self.confirmed_stop,
        )
        self.audio_events.append(event)
        self.operation_log.append(event)
        return event

    def push_chunk(self, chunk: Mapping[str, Any] | None = None, **kwargs: Any) -> Result:
        """Streaming vocabulary alias for :meth:`audio_arrive`."""
        if chunk is not None:
            if isinstance(chunk, Mapping):
                kwargs = {**dict(chunk), **kwargs}
            else:
                raise TypeError("chunk must be a mapping in this in-memory API")
        return self.audio_arrive(**kwargs)

    def arrive(self, *args: Any, **kwargs: Any) -> Result:
        return self.audio_arrive(*args, **kwargs)

    def confirm(self, confirmed_stop: int) -> Result:
        self.confirmed_stop = max(self.confirmed_stop, int(confirmed_stop))
        return Result(op="confirm", confirmed_stop=self.confirmed_stop)

    def seal_utterance(self, last_valid_frame: int | None = None) -> Result:
        self.sealed = True
        if last_valid_frame is not None:
            self.last_valid_frame = int(last_valid_frame)
            self.confirmed_stop = max(self.confirmed_stop, int(last_valid_frame) + 1)
        elif self.sentence is not None:
            lv = self.sentence.last_valid_frame(self.arrived_stop)
            self.last_valid_frame = lv
            if lv is not None:
                self.confirmed_stop = max(self.confirmed_stop, lv + 1)
        return Result(
            op="seal_utterance",
            sealed=True,
            last_valid_frame=self.last_valid_frame,
            confirmed_stop=self.confirmed_stop,
        )

    end_utterance = seal_utterance
    force_end = seal_utterance

    # ------------------------------------------------------------- constraints
    def add_annotations(self, records: Any) -> list[Result]:
        if isinstance(records, ConstraintSet):
            self.constraints = records
            return records.active_records()
        loaded = []
        for rec in records:
            loaded.append(self.constraints.load(rec))
        return loaded

    def reassess(self, word: str | int | None = None) -> Result:
        """Recompute training and online eligibility after a constraint edit."""
        index = self._index(word)
        suffix = self._suffix_feasible(index, self.last_committed_boundary)
        training = None
        if self.sentence is not None:
            from .training import assess_sentence

            training = assess_sentence(
                self.sentence,
                arrived_stop=self.arrived_stop,
                confirmed_stop=self.confirmed_stop,
                sealed=self.sealed,
            )
        proposal = None
        if index < len(self.word_ids):
            proposal = self.online_propose(index)
        return Result(
            word_index=index,
            suffix_feasible=suffix.feasible,
            suffix_reason=suffix.reason,
            training_eligibility=training,
            proposal=proposal,
        )

    def revise_annotation(self, revision: Mapping[str, Any]) -> Result:
        rid = str(revision.get("record", revision.get("id")))
        old = self.constraints.records.get(rid)
        word_index = old.word_index if old is not None else self.word_to_index.get(str(revision.get("word", "")))
        revised = self.constraints.revise(revision)
        committed_region = word_index in self.committed
        affected = self._invalidate_from(word_index) if word_index is not None else []
        version = Result(
            record_id=rid,
            revision=revised.revision,
            word_index=word_index,
            word=revised.word,
            committed_region=committed_region,
            existing_subtitle_immutable=committed_region,
            invalidated_proposals=affected,
            reassessment=None if word_index is None else self.reassess(word_index),
            constraint=revised,
        )
        self.interpretation_versions.append(version)
        self.operation_log.append(Result(op="revise_annotation", **version))
        return version

    def _invalidate_from(self, word_index: int | None) -> list[int]:
        if word_index is None:
            return []
        affected = [i for i in list(self.provisional) if i >= word_index]
        for i in affected:
            self.provisional[i].status = "stale_constraint"
            del self.provisional[i]
        return affected

    # ------------------------------------------------------------------- state
    @property
    def next_word_index(self) -> int:
        return len(self.committed_order)

    @property
    def last_committed_boundary(self) -> int:
        if not self.committed_order:
            return 0
        return int(self.committed[self.committed_order[-1]]["boundary"])

    def _index(self, word: str | int | None) -> int:
        if word is None:
            return self.next_word_index
        if isinstance(word, int):
            return word
        return self.word_to_index[str(word)]

    def _predecessor_boundary(self, index: int) -> tuple[int, str]:
        if index in self.committed:
            return int(self.committed[index]["boundary"]), "committed_self"
        if index == self.next_word_index:
            return self.last_committed_boundary, "last_committed"
        if index > 0 and index - 1 in self.provisional:
            prev_boundary = self.provisional[index - 1].get("boundary")
            if prev_boundary is not None:
                return int(prev_boundary), "provisional_predecessor"
        if index > 0 and index - 1 in self.committed:
            return int(self.committed[index - 1]["boundary"]), "committed_predecessor"
        return self.last_committed_boundary, "last_committed_fallback"

    @property
    def num_words(self) -> int:
        return len(self.word_ids)

    @property
    def num_frames(self) -> int:
        if self.sentence is not None:
            return self.sentence.num_frames
        if self.direct_num_frames is not None:
            return int(self.direct_num_frames)
        if self.direct_valid is not None:
            return len(self.direct_valid)
        return max(self.arrived_stop, 0)

    def selection_probability(self, index: int) -> np.ndarray:
        n = self.arrived_stop
        if self.direct_p is not None:
            arr = np.asarray(self.direct_p, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr[index, :n]
        if self.direct_e is not None:
            arr = np.asarray(self.direct_e, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            return sigmoid(arr[index, :n])
        if self.sentence is None:
            raise ValueError("selection probabilities require score arrays or a SentenceInputs")
        return sigmoid(self.sentence.selection_energy_row(index, self.arrived_stop))

    def chunk_probabilities_at(self, index: int, boundary: int, *, confirmed: bool) -> Result:
        lo, hi = window_range(boundary, self.chunk_width)
        if self.sentence is not None:
            u = self.sentence.chunk_energy_row(index, self.arrived_stop)
            v = self.sentence.V
            valid = self.sentence.valid[: self.arrived_stop]
        else:
            if self.direct_u is None or self.direct_V is None:
                zero = np.zeros(0)
                return Result(context=zero, weights=zero, attention_indices=np.arange(lo, hi + 1),
                              context_start=lo, context_stop=hi + 1, acoustic_range=[lo, hi + 1],
                              confirmed=confirmed)
            u_arr = np.asarray(self.direct_u, dtype=np.float64)
            u = u_arr[index] if u_arr.ndim == 2 else u_arr
            v = np.asarray(self.direct_V, dtype=np.float64)
            valid = (
                self.direct_valid[: self.arrived_stop]
                if self.direct_valid is not None
                else np.ones(self.arrived_stop, dtype=bool)
            )
        if confirmed:
            mask = np.asarray(valid, dtype=bool).copy()
            conf_mask = np.zeros(self.arrived_stop, dtype=bool)
            conf_mask[: min(self.arrived_stop, self.confirmed_stop)] = True
            mask &= conf_mask
        else:
            mask = np.asarray(valid, dtype=bool)
        wa = window_attention(u[: boundary + 1], boundary, self.chunk_width, mask[: boundary + 1])
        context = wa.weights @ v[wa.indices]
        used_indices = np.asarray(wa.indices)[np.asarray(wa.weights) > 0.0]
        return Result(
            context=context,
            attention=wa.weights,
            attention_indices=wa.indices,
            used_frame_indices=used_indices,
            context_start=lo,
            context_stop=hi + 1,
            acoustic_range=[lo, hi + 1],
            confirmed=confirmed,
        )

    def _window_confirmed(self, j: int) -> bool:
        valid = self.sentence.valid if self.sentence is not None else self.direct_valid
        if valid is not None and not bool(valid[j]):
            return False
        lo, hi = window_range(j, self.chunk_width)
        if hi >= self.confirmed_stop:
            return False
        if valid is not None:
            window_valid = np.flatnonzero(np.asarray(valid[lo : hi + 1], dtype=bool)) + lo
            if window_valid.size and int(window_valid.max()) >= self.confirmed_stop:
                return False
        return True

    def _constraint_factor(self, index: int, j: int) -> float:
        rows = self.constraints.active_records(index)
        factor = 1.0
        for rec in rows:
            if rec.lo <= j <= rec.hi:
                continue
            factor *= rec.outside_factor
        return float(factor)

    def _suffix_feasible(self, index: int, boundary: int) -> Result:
        n = self.num_frames
        c = self.constraints.factors_matrix(n, len(self.word_ids))
        suffix_c = c[index:]
        valid = self.sentence.valid if self.sentence is not None else self.direct_valid
        geo = geometric_feasibility(
            suffix_c,
            n,
            valid=valid,
            start_boundary=boundary,
            num_words=len(self.word_ids) - index,
        )
        return geo

    def soft_relationship(self, index: int, *, force_end: bool | None = None) -> Result:
        n = self.arrived_stop
        if self.direct_e is not None or self.direct_p is not None:
            e = None if self.direct_e is None else np.asarray(self.direct_e)[:, :n]
            p_override = None if self.direct_p is None else np.asarray(self.direct_p)
            if p_override is not None and p_override.ndim == 1:
                p_override = p_override[None, :]
            p_override = None if p_override is None else p_override[:, :n]
        elif self.sentence is not None:
            e = self.sentence.selection_energies(n)
            p_override = None
        else:
            return Result(available=False)
        c = self.constraints.factors_matrix(n, len(self.word_ids))
        do_force = self.sealed if force_end is None else force_end
        valid = self.sentence.valid if self.sentence is not None else self.direct_valid
        dist = boundary_distribution(
            None if e is None else e[index : index + 1],
            c[index : index + 1],
            p=None if p_override is None else p_override[index : index + 1],
            valid=None if valid is None else valid[:n],
            force_end=do_force,
            last_valid_frame=self.last_valid_frame,
        )
        alpha = dist.alpha
        if np.ndim(alpha) > 1:
            alpha = alpha[0]
        p = dist.p
        if np.ndim(p) > 1:
            p = p[0]
        return Result(
            available=True,
            alpha=alpha,
            boundary_probability=alpha,
            p=p,
            selection_probability=p,
            wait=float(np.atleast_1d(dist.wait)[0]),
            path_mass=dist.path_mass,
            tau=self.tau,
            probability_mass_above_tau_frames=float(alpha[p >= self.tau].sum()) if len(alpha) else 0.0,
        )

    # ---------------------------------------------------------------- propose
    def online_propose(
        self,
        word: str | int | None = None,
        *,
        temporary: bool = True,
        tau: float | None = None,
        max_lookahead: int | None = None,
        **options: Any,
    ) -> Proposal:
        if tau is None:
            tau = options.pop("threshold", None)
        if max_lookahead is None:
            max_lookahead = options.pop("lookahead", options.pop("L", None))
        index = self._index(word)
        if index >= len(self.word_ids):
            return Proposal(
                op="online_propose",
                word_index=index,
                word=None,
                status="all_words_committed",
                committable=False,
                temporary=False,
                boundary=None,
                reasons=["all_decoding_words_have_been_committed"],
                last_committed_boundary=self.last_committed_boundary,
            )
        tau = self.tau if tau is None else float(tau)
        L = self.max_lookahead if max_lookahead is None else int(max_lookahead)
        start, start_kind = self._predecessor_boundary(index)
        n = self.arrived_stop
        p = self.selection_probability(index) if n > 0 else np.zeros(0)

        reasons: list[str] = []
        formal_j: int | None = None
        formal_candidates: list[int] = []
        if n:
            upper = min(n, self.confirmed_stop)
            for j in range(max(0, start), upper):
                if not self._window_confirmed(j):
                    continue
                if p[j] >= tau:
                    formal_candidates.append(j)
                    formal_j = j
                    break

        preview_j: int | None = None
        likely_j: int | None = None
        preview_scope = "none"
        preview_upper = n
        if self.lookahead_reference == "confirmed_frontier":
            preview_upper = min(n, self.confirmed_stop + L)
        elif self.lookahead_reference == "acquired_end":
            # L is a trailing-distance window into the current acquisition end.
            preview_upper = n
            preview_lower = max(start, n - L)
        elif self.lookahead_reference == "last_boundary":
            preview_upper = min(n, start + L + 1)
        else:
            preview_lower = start
        preview_lower = locals().get("preview_lower", start)
        if temporary and n and formal_j is None:
            candidates = []
            valid = self.sentence.valid if self.sentence is not None else self.direct_valid
            for j in range(max(0, preview_lower), preview_upper):
                if valid is not None and not bool(valid[j]):
                    continue
                if j < start:
                    continue
                candidates.append(j)
            crossed = [j for j in candidates if p[j] >= tau]
            if crossed:
                preview_j = crossed[0]
                preview_scope = "first_threshold_in_preview"
            elif candidates:
                likely_j = int(max(candidates, key=lambda j: p[j]))
                preview_scope = "argmax_selection_in_preview"

        chosen = formal_j if formal_j is not None else preview_j
        is_temporary = formal_j is None and preview_j is not None
        if chosen is None:
            status = "waiting"
            reasons.append("no_threshold_boundary_available")
            if self.confirmed_stop < n:
                reasons.append("audio_not_confirmed")
            if n == 0:
                reasons.append("no_audio_arrived")
            likely_context = None
            likely_range = None
            if likely_j is not None:
                likely_context = self.chunk_probabilities_at(index, likely_j, confirmed=False)
                likely_range = likely_context.acoustic_range
                reasons.append("temporary_most_likely_context_available")
            proposal = Proposal(
                op="online_propose",
                word_index=index,
                word=self.word_ids[index] if index < len(self.word_ids) else str(index),
                status=status,
                committable=False,
                temporary=False,
                preview_available=likely_j is not None,
                temporary_context_available=likely_j is not None,
                boundary=None,
                selected_boundary=None,
                most_likely_boundary=likely_j,
                temporary_boundary=likely_j,
                preview_boundary=likely_j,
                temporary_context=None if likely_context is None else likely_context.context,
                acoustic_range=likely_range,
                start_boundary=start,
                start_kind=start_kind,
                tau=tau,
                max_lookahead=L,
                reasons=reasons,
                arrived_stop=n,
                confirmed_stop=self.confirmed_stop,
            )
            self.provisional[index] = proposal
            return proposal

        factor = self._constraint_factor(index, chosen)
        constraint_ok = factor > 0.0
        suffix = self._suffix_feasible(index, chosen)
        window_ok = self._window_confirmed(chosen) if not is_temporary else True
        sequence_ok = index == self.next_word_index or index in self.provisional or index in self.committed
        monotone_ok = chosen >= self.last_committed_boundary
        committable = (
            not is_temporary
            and constraint_ok
            and bool(suffix.feasible)
            and window_ok
            and sequence_ok
            and monotone_ok
        )
        if is_temporary:
            status = "provisional"
            reasons.append("boundary_uses_unconfirmed_audio")
        elif not constraint_ok:
            status = "constraint_blocked"
            reasons.append("hard_time_window_excludes_boundary")
        elif not suffix.feasible:
            status = "constraints_infeasible"
            reasons.append(str(suffix.reason))
        elif not window_ok:
            status = "waiting"
            reasons.append("chunk_window_not_confirmed")
        elif not sequence_ok:
            status = "waiting_predecessor"
            reasons.append("word_sequence_gap")
        else:
            status = "committable"
            reasons.append("confirmed_threshold_and_window")

        ctx_info = self.chunk_probabilities_at(index, chosen, confirmed=not is_temporary)
        rel = self.soft_relationship(index)
        distance_to_end = max(0, n - 1 - chosen)
        exclusive_distance_to_end = n - chosen
        proposal = Proposal(
            op="online_propose",
            word_index=index,
            word=self.word_ids[index] if index < len(self.word_ids) else str(index),
            status=status,
            committable=committable,
            temporary=is_temporary,
            preview=is_temporary,
            boundary=int(chosen),
            selected_boundary=int(chosen),
            temporary_boundary=int(chosen) if is_temporary else None,
            most_likely_boundary=int(likely_j if likely_j is not None else chosen),
            p=float(p[chosen]) if n else None,
            selection_probability=float(p[chosen]) if n else None,
            constraint_factor=factor,
            constraint_ok=constraint_ok,
            suffix_feasible=bool(suffix.feasible),
            suffix_reason=suffix.reason,
            window_confirmed=window_ok,
            start_boundary=start,
            start_kind=start_kind,
            tau=tau,
            max_lookahead=L,
            lookahead_reference=self.lookahead_reference,
            distance_to_acquired_end=distance_to_end,
            exclusive_distance_to_acquired_end=exclusive_distance_to_end,
            within_lookahead_distance=exclusive_distance_to_end <= L,
            backfill=formal_j is not None and exclusive_distance_to_end > L,
            latency_tier_compliant=exclusive_distance_to_end <= L,
            preview_scope=preview_scope if is_temporary else "confirmed",
            context=ctx_info.context,
            acoustic_range=ctx_info.acoustic_range,
            context_start=ctx_info.context_start,
            context_stop=ctx_info.context_stop,
            attention_indices=ctx_info.attention_indices,
            used_frame_indices=ctx_info.used_frame_indices,
            attention=ctx_info.attention,
            soft_boundary_probability=(
                float(rel.alpha[chosen]) if rel.available and chosen < len(rel.alpha) else None
            ),
            soft_mode_boundary=(
                int(np.argmax(rel.alpha))
                if rel.available and hasattr(rel.alpha, "__len__") and len(rel.alpha)
                else None
            ),
            soft_wait=rel.wait if rel.available else None,
            wait=rel.wait if rel.available else None,
            waiting_mass=rel.wait if rel.available else None,
            soft_path_mass=rel.path_mass if rel.available else None,
            reasons=reasons,
            arrived_stop=n,
            confirmed_stop=self.confirmed_stop,
        )
        # Do not overwrite a formal proposal accidentally; store disposable.
        self.provisional[index] = proposal
        self.operation_log.append(proposal)
        return proposal

    def temporary_query(self, word: str | int | None = None) -> Proposal:
        """Return the disposable most-likely context even below tau."""
        result = self.online_propose(word)
        if result.get("boundary") is None and result.get("most_likely_boundary") is not None:
            j = int(result.most_likely_boundary)
            ctx = self.chunk_probabilities_at(self._index(word), j, confirmed=False)
            result["preview_boundary"] = j
            result["temporary_boundary"] = j
            result["preview_context"] = ctx.context
            result["preview_acoustic_range"] = ctx.acoustic_range
        return result

    preview = temporary_query

    # ------------------------------------------------------------------ commit
    def commit(
        self,
        word: str | int | None = None,
        boundary: int | None = None,
        *,
        proposal: Proposal | Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> Result:
        if isinstance(word, Mapping) and proposal is None:
            proposal = word
            word = proposal.get("word_index", proposal.get("word"))
        index = self._index(word)
        if index >= len(self.word_ids):
            return Result(op="commit", accepted=False, reason="all_words_committed", word_index=index)
        if index in self.committed:
            existing = self.committed[index]
            requested = boundary
            if requested is None and isinstance(proposal, Mapping):
                requested = proposal.get("boundary")
            if requested is None or int(requested) == int(existing.boundary):
                return Result(
                    op="commit",
                    accepted=True,
                    idempotent=True,
                    immutable=True,
                    word_index=index,
                    boundary=int(existing.boundary),
                    record=existing,
                )
            return Result(
                op="commit",
                accepted=False,
                immutable=True,
                reason="committed_subtitle_boundary_cannot_move",
                existing_boundary=int(existing.boundary),
                requested_boundary=int(requested),
            )
        if proposal is None and index in self.provisional:
            proposal = self.provisional[index]
        if boundary is not None:
            # Re-evaluate at caller-supplied boundary.
            prop = self.online_propose(word or index)
            chosen = int(boundary)
            proposal = Proposal(prop)
            proposal.boundary = chosen
            proposal.temporary = chosen >= self.confirmed_stop
            proposal.committable = False  # rechecked below
        elif proposal is not None and proposal.get("boundary") is not None:
            chosen = int(proposal["boundary"])
        else:
            evaluated = self.online_propose(index)
            proposal = evaluated
            if evaluated.boundary is None:
                return Result(op="commit", accepted=False, word_index=index, reasons=evaluated.reasons)
            chosen = int(evaluated.boundary)

        reasons: list[str] = []
        sequence_ok = index == self.next_word_index
        monotone_ok = chosen >= self.last_committed_boundary
        window_ok = self._window_confirmed(chosen)
        factor = self._constraint_factor(index, chosen)
        suffix = self._suffix_feasible(index, chosen)
        if not sequence_ok:
            reasons.append("word_index_is_not_next_continuous_word")
        if not monotone_ok:
            reasons.append("boundary_moves_committed_cursor_left")
        if not window_ok:
            reasons.append("chunk_window_is_not_confirmed")
        if factor <= 0.0:
            reasons.append("hard_constraint_excludes_boundary")
        if not suffix.feasible:
            reasons.append(f"joint_path_{suffix.reason}")

        accepted = force or not reasons
        lo, hi_inclusive = window_range(chosen, self.chunk_width)
        receipt = Result(
            op="commit",
            accepted=accepted,
            word_index=index,
            word=self.word_ids[index] if index < len(self.word_ids) else str(index),
            boundary=chosen,
            acoustic_range=[lo, hi_inclusive + 1],
            reasons=reasons,
            immutable=accepted,
        )
        if not accepted:
            self.operation_log.append(receipt)
            return receipt

        hi = hi_inclusive
        # Freeze the exact context vector and window weights at acceptance.
        # Later array mutation/revision cannot silently alter this subtitle.
        frozen = self.chunk_probabilities_at(index, chosen, confirmed=True)
        record = Result(
            word_index=index,
            word=receipt.word,
            boundary=chosen,
            context=np.array(frozen.context, copy=True),
            context_vector=np.array(frozen.context, copy=True),
            window_attention=np.array(frozen.attention, copy=True),
            used_frame_indices=np.array(frozen.used_frame_indices, copy=True),
            context_start=lo,
            context_stop=hi + 1,
            acoustic_range=[lo, hi + 1],
            context_confirmed=True,
            revision_of_constraints={r.id: r.revision for r in self.constraints.active_records(index)},
        )
        self.committed[index] = record
        if index not in self.committed_order:
            self.committed_order.append(index)
        self.committed_order.sort()
        self.subtitles.append(record)
        self.provisional.pop(index, None)
        # Later provisional words must be re-read from the new formal cursor.
        for later in [i for i in list(self.provisional) if i > index]:
            self.provisional[later].status = "stale_after_commit"
            del self.provisional[later]
        self.operation_log.append(receipt)
        return receipt

    def confirm_commit(self, *args: Any, **kwargs: Any) -> Result:
        """Alias for callers using the event vocabulary."""
        return self.commit(*args, **kwargs)

    def withdraw_uncommitted(self, word: str | int | Mapping[str, Any] | None = None) -> Result:
        if isinstance(word, Mapping):
            word = word.get("word_index", word.get("word"))
        index = self._index(word)
        if index in self.committed:
            receipt = Result(
                op="withdraw_uncommitted",
                accepted=False,
                word_index=index,
                reason="committed_subtitle_is_immutable",
                last_committed_boundary=self.last_committed_boundary,
            )
            self.operation_log.append(receipt)
            return receipt
        removed = [i for i in list(self.provisional) if i >= index]
        for i in removed:
            self.provisional.pop(i, None)
        receipt = Result(
            op="withdraw_uncommitted",
            accepted=True,
            word_index=index,
            removed_provisional=removed,
            last_committed_boundary=self.last_committed_boundary,
            next_read_boundary=self.last_committed_boundary,
        )
        self.operation_log.append(receipt)
        return receipt

    def get_state(self) -> Result:
        return Result(
            arrived_stop=self.arrived_stop,
            confirmed_stop=self.confirmed_stop,
            sealed=self.sealed,
            last_committed_boundary=self.last_committed_boundary,
            next_word_index=self.next_word_index,
            committed=[Result(r) for r in self.committed.values()],
            provisional=dict(self.provisional),
            subtitles=self.current_subtitles(),
        )

    def current_subtitles(self) -> list[Result]:
        return [Result(r) for r in self.subtitles]

    # Common vocabulary aliases.
    propose = online_propose
    propose_subtitle = online_propose
    request_proposal = online_propose
    read = online_propose
    arrive_audio = audio_arrive
    add_audio = audio_arrive
    confirm_audio = confirm
    commit_word = commit
    commit_subtitle = commit
    confirm_subtitle = commit
    withdraw = withdraw_uncommitted
    withdraw_uncommitted_word = withdraw_uncommitted
    revise_constraint = revise_annotation
    revise_constraints = revise_annotation
    update_annotation = revise_annotation
