"""Stateful thin wrapper around the stateless training forward/backward core."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .result import Result
from .sentence import SentenceInputs
from .training import train_forward_backward, train_sentence


class StreamingTrainer:
    """Accept acoustic chunks and run training on the currently available range.

    The class stores arrival/confirmation cursors and an optional reference to
    sentence arrays.  It does not enumerate paths or keep a history of score
    matrices; callers control whether the returned gradients are retained.
    """

    def __init__(
        self,
        sentence: SentenceInputs | Mapping[str, Any] | None = None,
        *,
        e: np.ndarray | None = None,
        u: np.ndarray | None = None,
        V: np.ndarray | None = None,
        chunk_width: int | None = None,
        constraints: Any = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(sentence, Mapping):
            sentence = SentenceInputs.from_materials(sentence)
        self.sentence = sentence
        self.e = e
        self.u = u
        self.V = V
        self.chunk_width = chunk_width or (sentence.chunk_width if sentence else None)
        self.constraints = constraints if constraints is not None else (sentence.constraints if sentence else None)
        self.kwargs = kwargs
        self.arrived_stop = 0
        self.confirmed_stop = 0
        self.sealed = False
        self.last_result: Result | None = None

    def push_chunk(
        self,
        start: int | None = None,
        stop: int | None = None,
        confirmed_stop: int | None = None,
        chunk: Mapping[str, Any] | None = None,
    ) -> Result:
        if chunk is not None:
            start = chunk.get("start", start)
            stop = chunk.get("stop", stop)
            confirmed_stop = chunk.get("confirmed_stop", confirmed_stop)
        if stop is not None:
            self.arrived_stop = max(self.arrived_stop, int(stop))
        if confirmed_stop is not None:
            self.confirmed_stop = max(self.confirmed_stop, int(confirmed_stop))
        return Result(arrived_stop=self.arrived_stop, confirmed_stop=self.confirmed_stop)

    audio_arrive = push_chunk

    def seal(self, last_valid_frame: int | None = None) -> Result:
        self.sealed = True
        self.last_valid_frame = last_valid_frame
        return Result(sealed=True, last_valid_frame=last_valid_frame)

    def forward_backward(self, **overrides: Any) -> Result:
        kwargs = dict(self.kwargs)
        kwargs.update(overrides)
        if self.sentence is not None and self.e is None:
            kwargs.setdefault("arrived_stop", self.arrived_stop or None)
            kwargs.setdefault("sealed", self.sealed)
            return train_sentence(self.sentence, **kwargs)

        if self.e is None or self.u is None or self.V is None or self.chunk_width is None:
            raise ValueError("direct StreamingTrainer requires e, u, V, and chunk_width")
        result = train_forward_backward(
            self.e,
            self.u,
            self.V,
            int(self.chunk_width),
            constraints=self.constraints,
            stop=self.arrived_stop or None,
            force_end=bool(kwargs.pop("force_end", self.sealed)),
            **kwargs,
        )
        self.last_result = result
        return result

    step = forward_backward
