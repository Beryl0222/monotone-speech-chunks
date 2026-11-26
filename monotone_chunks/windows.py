"""Chunk-window attention and exact context vector-Jacobian products."""
from __future__ import annotations

import numpy as np

from .result import Result


def window_range(boundary: int, chunk_width: int) -> tuple[int, int]:
    """Return inclusive ``[lo, j]`` from the public contract."""
    j = int(boundary)
    lo = max(0, j - int(chunk_width) + 1)
    return lo, j


def window_attention(
    u: np.ndarray,
    boundary: int,
    chunk_width: int,
    frame_mask: np.ndarray | None = None,
) -> Result:
    """Softmax u over ``[max(0,j-chunk_width+1), j]``."""
    scores = np.asarray(u, dtype=np.float64)
    n = scores.shape[0]
    if not 0 <= int(boundary) < n:
        raise IndexError("boundary is outside the score range")
    lo, j = window_range(boundary, chunk_width)
    idx = np.arange(lo, j + 1)
    vals = scores[idx]
    if frame_mask is not None:
        mask = np.asarray(frame_mask, dtype=bool)[idx]
        vals = np.where(mask, vals, -np.inf)
    if not np.any(np.isfinite(vals)):
        weights = np.zeros_like(vals)
    else:
        m = np.max(vals[np.isfinite(vals)])
        exp = np.exp(np.where(np.isfinite(vals), vals - m, 0.0))
        exp = exp * np.isfinite(vals)
        weights = exp / exp.sum()
    return Result(weights=weights, indices=idx, start=lo, stop=j + 1)


def window_contexts(
    u: np.ndarray,
    V: np.ndarray,
    chunk_width: int,
    frame_mask: np.ndarray | None = None,
) -> Result:
    """All boundary window contexts without materializing W x T x cw x D."""
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(V, dtype=np.float64)
    n = u_arr.shape[0]
    if v_arr.shape[0] < n:
        raise ValueError("V is shorter than u")
    if frame_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(frame_mask, dtype=bool)[:n]

    width = min(int(chunk_width), n) if n else int(chunk_width)
    att = np.zeros((n, max(int(chunk_width), 0)), dtype=np.float64)
    if n == 0:
        return Result(attention=att, contexts=np.zeros((0, v_arr.shape[1])), width=chunk_width)

    logits = np.full((n, int(chunk_width)), -np.inf, dtype=np.float64)
    for r in range(min(int(chunk_width), n)):
        j_idx = np.arange(r, n)
        k_idx = j_idx - r
        eligible = mask[k_idx]
        if np.any(eligible):
            logits[j_idx[eligible], r] = u_arr[k_idx[eligible]]

    finite = np.isfinite(logits)
    row_max = np.full(n, 0.0, dtype=np.float64)
    has = finite.any(axis=1)
    if np.any(has):
        row_max[has] = np.max(logits[has], axis=1)
    exp = np.zeros_like(logits)
    exp[finite] = np.exp(logits[finite] - np.broadcast_to(row_max[:, None], logits.shape)[finite])
    denom = exp.sum(axis=1)
    good = denom > 0.0
    att[good] = exp[good] / denom[good, None]

    contexts = np.zeros((n, v_arr.shape[1]), dtype=np.float64)
    for r in range(min(int(chunk_width), n)):
        j_idx = np.arange(r, n)
        k_idx = j_idx - r
        a = att[j_idx, r]
        contexts[j_idx] += a[:, None] * v_arr[k_idx]
    return Result(attention=att, contexts=contexts, width=int(chunk_width))


def context_forward_backward(
    u: np.ndarray,
    V: np.ndarray,
    chunk_width: int,
    alpha: np.ndarray | None = None,
    context_upstream: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
) -> Result:
    """Compute alpha-averaged context and VJP wrt alpha/u/V.

    If ``alpha is None`` this returns the matrix of boundary-specific window
    contexts.  The gradient formulas assign each contribution only to acoustic
    frames that actually belong to the corresponding boundary window.
    """
    wc_result = window_contexts(u, V, chunk_width, frame_mask)
    wc = wc_result.contexts
    att = wc_result.attention
    n = wc.shape[0]
    v_arr = np.asarray(V, dtype=np.float64)[:n]
    if alpha is None:
        context = wc
        alpha_arr = None
    else:
        alpha_arr = np.asarray(alpha, dtype=np.float64)[:n]
        context = alpha_arr @ wc

    out = Result(window_contexts=wc, attention=att, context=context)
    if context_upstream is None:
        return out

    g = np.asarray(context_upstream, dtype=np.float64)
    if g.shape != (v_arr.shape[1],):
        raise ValueError(f"context_upstream must have shape ({v_arr.shape[1]},)")
    if alpha_arr is None:
        raise ValueError("alpha is required when context_upstream is supplied")

    grad_alpha = wc @ g
    dot_v = v_arr @ g
    grad_u = np.zeros(n, dtype=np.float64)
    grad_v = np.zeros_like(v_arr)
    cw = int(chunk_width)
    for r in range(min(cw, n)):
        j_idx = np.arange(r, n)
        k_idx = j_idx - r
        a = att[j_idx, r]
        scale = alpha_arr[j_idx] * a
        # g . window_context_j = grad_alpha[j]
        grad_u[k_idx] += scale * (dot_v[k_idx] - grad_alpha[j_idx])
        grad_v[k_idx] += scale[:, None] * g[None, :]
    out.update(grad_alpha=grad_alpha, grad_u=grad_u, grad_v=grad_v)
    return out
