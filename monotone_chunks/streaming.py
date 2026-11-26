"""Streaming reader submodule alias."""
from .reader import Proposal, StreamingReader

OnlineReader = StreamingReader
StreamingSession = StreamingReader
MonotoneChunkReader = StreamingReader

__all__ = [
    "Proposal",
    "StreamingReader",
    "OnlineReader",
    "StreamingSession",
    "MonotoneChunkReader",
]
