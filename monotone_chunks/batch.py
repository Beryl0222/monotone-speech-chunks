"""Batched tensor helpers.

Batching is a loop over independent sentence workspaces.  In particular, window
softmax never sees a frame from another batch item; padding masks are passed
through to each item.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .result import Result
from .training import train_forward_backward
from .boundary import boundary_distribution


def boundary_distribution_batch(e, constraints=None, valid=None, **kwargs):
    """Independent per-item boundary DP; padded items never share state."""
    e_arr = np.asarray(e, dtype=np.float64)
    if e_arr.ndim != 3:
        raise ValueError("batched e must have shape (batch, words, frames)")
    b, w, t = e_arr.shape
    alphas = np.zeros_like(e_arr)
    waits = np.zeros((b, w), dtype=np.float64)
    masses = np.zeros(b, dtype=np.float64)
    for bi in range(b):
        c_b = None if constraints is None else _item(constraints, bi)
        v_b = None if valid is None else np.asarray(valid, dtype=bool)[bi]
        out = boundary_distribution(e_arr[bi], c_b, valid=v_b, **kwargs)
        alphas[bi] = out.alpha
        waits[bi] = np.atleast_1d(out.wait)
        masses[bi] = out.path_mass
    return Result(alpha=alphas, boundary_probability=alphas, wait=waits, path_mass=masses)


def _item(x: Any, b: int) -> Any:
    if x is None:
        return None
    arr = np.asarray(x)
    return arr[b] if arr.ndim >= 2 and arr.shape[0] > b and arr.dtype != object else x


def train_tensor_batch(
    e: np.ndarray,
    u: np.ndarray,
    V: np.ndarray,
    chunk_width: int,
    *,
    constraints: np.ndarray | Sequence[Any] | None = None,
    valid: np.ndarray | None = None,
    context_upstream: np.ndarray | None = None,
    sample_weights: np.ndarray | Sequence[float] | None = None,
    eligible: Sequence[bool] | None = None,
    **kwargs: Any,
) -> Result:
    e_arr = np.asarray(e, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(V, dtype=np.float64)
    b_size = e_arr.shape[0]
    if v_arr.ndim == 2:
        v_arr = np.broadcast_to(v_arr, (b_size, *v_arr.shape)).copy()
    if valid is not None:
        valid_arr = np.asarray(valid, dtype=bool)
    else:
        valid_arr = np.ones((b_size, e_arr.shape[-1]), dtype=bool)
    if context_upstream is not None:
        gu_arr = np.asarray(context_upstream, dtype=np.float64)
    else:
        gu_arr = None
    if sample_weights is None:
        weights = np.ones(b_size, dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)

    reports = []
    denominator = 0.0
    for b in range(b_size):
        is_eligible = True if eligible is None else bool(eligible[b])
        if not is_eligible:
            reports.append(Result(eligible=False, status="excluded_by_caller"))
            continue
        denominator += float(weights[b])
        c_b = None if constraints is None else _item(constraints, b)
        gu_b = None if gu_arr is None else gu_arr[b]
        rep = train_forward_backward(
            e_arr[b],
            u_arr[b],
            v_arr[b],
            chunk_width,
            constraints=c_b,
            valid=valid_arr[b],
            context_upstream=gu_b,
            **kwargs,
        )
        rep.sample_weight = float(weights[b])
        reports.append(rep)

    loss = 0.0
    grad_keys = ["grad_e", "grad_u", "grad_V"]
    stacked: dict[str, list[np.ndarray]] = {k: [] for k in grad_keys}
    for b, rep in enumerate(reports):
        if not rep.get("eligible", True) or "loss" not in rep:
            continue
        scale = float(weights[b]) / denominator if denominator else 0.0
        loss += scale * rep.loss
        for k in grad_keys:
            if k in rep:
                stacked[k].append(scale * rep[k])
    grads = {k: np.stack(v) for k, v in stacked.items() if v}
    return Result(
        loss=loss,
        eligible_weight_denominator=denominator,
        sentence_weight_denominator=denominator,
        reports=reports,
        **grads,
    )
