"""Compatibility facade for tests/tools that import the project as ``solution``."""
from monotone_chunks import *  # noqa: F403
from monotone_chunks import (  # noqa: F401
    api,
    batch,
    boundary,
    constraints,
    reader,
    replay,
    sentence,
    training,
    windows,
)
from monotone_chunks.api import *  # noqa: F403

__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    from monotone_chunks.replay import main

    raise SystemExit(main())
