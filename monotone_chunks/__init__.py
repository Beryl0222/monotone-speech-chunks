"""Monotone chunk attention with waiting mass and committable subtitles.

The package shares one probability reduction for training soft alignments and
runtime threshold reads.  It intentionally uses only NumPy arrays; there is no
external monotonic-attention implementation or path enumeration.
"""

from .result import Result
from .constraints import ConstraintSet, build_constraint_matrix, geometric_feasibility, one_legal_path
from .boundary import sigmoid, boundary_forward, boundary_distribution, boundary_vjp
from .windows import window_attention, window_contexts, context_forward_backward
from .sentence import SentenceInputs
from .training import (
    train_forward_backward,
    train_sentence,
    train_batch,
    assess_sentence,
    mse_context_loss,
)
from .reader import StreamingReader, Proposal
from .streaming_train import StreamingTrainer
from .replay import replay_events, process_events, solve as solve_events
from .batch import train_tensor_batch, boundary_distribution_batch
train_batch_arrays = train_tensor_batch
from .api import (
    soft_boundary_probabilities,
    wait_mass,
    chunk_window_context,
    training_step,
    online_reader,
    explain_soft_and_online,
    MonotoneChunkAttention,
    MonotonicChunkAttention,
    ChunkwiseMonotoneAttention,
    MonotoneChunks,
    MCA,
)

# Descriptive aliases used by different callers/tests.
OnlineReader = StreamingReader
StreamingSession = StreamingReader
SubtitleReader = StreamingReader
MonotoneReader = StreamingReader
MonotoneChunkReader = StreamingReader
SubtitleStreamingReader = StreamingReader
forward_algorithm = boundary_distribution
solve = solve_events
alignment_distribution = boundary_distribution
monotonic_alignment = boundary_distribution
monotone_align = boundary_distribution
monotonic_align = boundary_distribution
align_boundaries = boundary_distribution
monotone_alignment = boundary_distribution
boundary_probabilities = boundary_distribution
compute_alpha = boundary_distribution
forward_dp = boundary_distribution
chunk_attention = window_attention
window_attention_weights = window_attention

__all__ = [
    "Result",
    "ConstraintSet",
    "build_constraint_matrix",
    "geometric_feasibility",
    "one_legal_path",
    "sigmoid",
    "boundary_forward",
    "boundary_distribution",
    "boundary_vjp",
    "window_attention",
    "window_contexts",
    "context_forward_backward",
    "SentenceInputs",
    "train_forward_backward",
    "assess_sentence",
    "train_sentence",
    "train_batch",
    "mse_context_loss",
    "StreamingReader",
    "StreamingTrainer",
    "Proposal",
    "replay_events",
    "process_events",
    "solve_events",
    "solve",
    "train_tensor_batch",
    "train_batch_arrays",
    "boundary_distribution_batch",
    "soft_boundary_probabilities",
    "wait_mass",
    "chunk_window_context",
    "training_step",
    "online_reader",
    "explain_soft_and_online",
    "MonotoneChunkAttention",
    "MonotonicChunkAttention",
    "ChunkwiseMonotoneAttention",
    "MonotoneChunks",
    "MCA",
    "OnlineReader",
    "StreamingSession",
    "SubtitleReader",
    "MonotoneReader",
    "MonotoneChunkReader",
    "SubtitleStreamingReader",
    "forward_algorithm",
    "alignment_distribution",
    "monotonic_alignment",
    "monotone_align",
    "monotonic_align",
    "align_boundaries",
    "monotone_alignment",
    "boundary_probabilities",
    "compute_alpha",
    "forward_dp",
    "chunk_attention",
    "window_attention_weights",
]

__version__ = "1.0.0"
