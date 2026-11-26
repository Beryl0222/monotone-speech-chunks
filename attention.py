from monotone_chunks.windows import (
    context_forward_backward,
    window_attention,
    window_contexts,
)

chunk_attention = window_attention
chunk_contexts = window_contexts

__all__ = [
    "window_attention",
    "window_contexts",
    "context_forward_backward",
    "chunk_attention",
    "chunk_contexts",
]
