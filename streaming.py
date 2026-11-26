"""Streaming reader compatibility module."""
from monotone_chunks.reader import Proposal, StreamingReader

OnlineReader = StreamingReader
StreamingSession = StreamingReader
MonotoneChunkReader = StreamingReader
SubtitleReader = StreamingReader

__all__ = [
    "Proposal",
    "StreamingReader",
    "OnlineReader",
    "StreamingSession",
    "MonotoneChunkReader",
    "SubtitleReader",
]
