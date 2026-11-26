"""Small descriptive facade over the project's mathematical core."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .boundary import boundary_distribution
from .reader import StreamingReader
from .sentence import SentenceInputs
from .training import train_forward_backward
from .windows import window_attention, window_contexts


def soft_boundary_probabilities(e: np.ndarray, constraints: Any = None, **kwargs: Any):
    """Constrained training soft distribution alpha plus waiting mass."""
    return boundary_distribution(e, constraints, **kwargs)


def wait_mass(e: np.ndarray, constraints: Any = None, **kwargs: Any) -> np.ndarray:
    """Return only the residual block-end waiting mass."""
    return boundary_distribution(e, constraints, **kwargs).wait


def chunk_window_context(u: np.ndarray, V: np.ndarray, chunk_width: int, **kwargs: Any):
    """Boundary-specific chunk softmax contexts (not alpha averaged)."""
    return window_contexts(u, V, chunk_width, **kwargs)


def training_step(*args: Any, **kwargs: Any):
    """Forward/backward training entry point."""
    return train_forward_backward(*args, **kwargs)


def online_reader(
    materials_or_sentence: Mapping[str, Any] | SentenceInputs | None = None,
    **kwargs: Any,
) -> StreamingReader:
    if isinstance(materials_or_sentence, SentenceInputs):
        return StreamingReader(materials_or_sentence, **kwargs)
    if materials_or_sentence is not None:
        return StreamingReader.from_materials(materials_or_sentence, **kwargs)
    return StreamingReader(**kwargs)


def explain_soft_and_online(reader: StreamingReader, word: str | int | None = None) -> dict[str, Any]:
    """Join the soft distribution, threshold decision, and commit reason."""
    index = reader._index(word)
    proposal = reader.online_propose(index)
    relationship = reader.soft_relationship(index)
    return {
        "word_index": index,
        "word": reader.word_ids[index],
        "online": proposal,
        "soft": relationship,
        "relationship": {
            "tau": reader.tau,
            "online_boundary": proposal.get("boundary"),
            "selection_probability": proposal.get("selection_probability"),
            "soft_alpha_at_online_boundary": proposal.get("soft_boundary_probability"),
            "soft_wait": proposal.get("soft_wait"),
            "note": "soft DP mass and hard threshold crossings are related but not numerically identical algorithms",
        },
    }


class MonotoneChunkAttention:
    """Small object-oriented facade for callers that prefer a model object.

    The mathematical implementation remains the functional NumPy core; this
    class only binds common chunk/tier configuration.
    """

    def __init__(
        self,
        chunk_width: int | None = None,
        *,
        tau: float = 0.5,
        max_lookahead: int = 3,
        constraints: Any = None,
        **kwargs: Any,
    ) -> None:
        self.chunk_width = chunk_width if chunk_width is not None else kwargs.get("chunk_size", 1)
        self.tau = tau
        self.max_lookahead = max_lookahead
        self.constraints = constraints
        self.config = kwargs

    def boundary_distribution(self, e: np.ndarray, constraints: Any = None, **kwargs: Any):
        return boundary_distribution(
            e,
            self.constraints if constraints is None else constraints,
            **kwargs,
        )

    forward_alignment = boundary_distribution

    def window_contexts(self, u: np.ndarray, V: np.ndarray, chunk_width: int | None = None, **kwargs: Any):
        return window_contexts(u, V, chunk_width or self.chunk_width, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("chunk_width", self.chunk_width)
        kwargs.setdefault("constraints", self.constraints)
        return train_forward_backward(*args, **kwargs)

    forward_backward = forward
    training_step = forward

    def reader(self, sentence: SentenceInputs | Mapping[str, Any] | None = None, **kwargs: Any) -> StreamingReader:
        kwargs.setdefault("chunk_width", self.chunk_width)
        kwargs.setdefault("tau", self.tau)
        kwargs.setdefault("max_lookahead", self.max_lookahead)
        kwargs.setdefault("constraints", self.constraints)
        return online_reader(sentence, **kwargs)

    streaming_reader = reader
    subtitle_reader = reader


MonotonicChunkAttention = MonotoneChunkAttention
ChunkwiseMonotoneAttention = MonotoneChunkAttention
MonotoneChunks = MonotoneChunkAttention
MCA = MonotoneChunkAttention

