"""Dictionary result with attribute access used throughout the project."""
from __future__ import annotations

from typing import Any


class Result(dict):
    """A plain JSON-friendly mapping with attribute access.

    Nested dictionaries produced by user code are not automatically converted;
    construct ``Result`` explicitly for objects whose attributes are needed.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:  # pragma: no cover - trivial
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
