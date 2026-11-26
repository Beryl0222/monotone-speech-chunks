"""Soft monotone boundary reduction and its reverse-mode derivative."""
from __future__ import annotations

from typing import Any

import numpy as np

from .constraints import ConstraintSet, build_constraint_matrix
from .result import Result


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    x_arr = np.asarray(x, dtype=np.float64)
    positive = x_arr >= 0
    out = np.empty_like(x_arr)
    exp_neg = np.exp(-x_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(x_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out if x_arr.ndim else float(out)


def compose_frame_mask(
    num_frames: int,
    *,
    valid: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
    received_stop: int | None = None,
    stop: int | None = None,
) -> np.ndarray:
    n = int(num_frames)
    if frame_mask is not None:
        mask = np.asarray(frame_mask, dtype=bool)
        if mask.shape[0] < n:
            raise ValueError("frame_mask is shorter than the requested frame range")
        return mask[:n].copy()
    mask = np.ones(n, dtype=bool)
    if valid is not None:
        mask &= np.asarray(valid[:n], dtype=bool)
    if received_stop is not None:
        received = np.zeros(n, dtype=bool)
        received[: min(n, int(received_stop))] = True
        mask &= received
    elif stop is not None:
        mask &= np.arange(n) < int(stop)
    return mask


def _as_2d(e: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(e, dtype=np.float64)
    squeezed = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeezed = True
    if arr.ndim != 2:
        raise ValueError(f"e must be 1D or 2D, got shape {arr.shape}")
    return arr, squeezed


def _constraint_array(
    constraints: Any,
    w: int,
    n: int,
    word_to_index: dict[str, int] | None = None,
) -> np.ndarray:
    if constraints is None:
        return np.ones((w, n), dtype=np.float64)
    if isinstance(constraints, ConstraintSet):
        return constraints.factors_matrix(n, w)
    if isinstance(constraints, np.ndarray):
        if constraints.ndim == 1:
            if w != 1:
                raise ValueError("1D constraint vector only valid for one word")
            return constraints[None, :n].astype(np.float64, copy=False)
        return build_constraint_matrix(w, n, constraints)
    return build_constraint_matrix(w, n, constraints, word_to_index)


def row_forward(
    p: np.ndarray,
    constraint: np.ndarray,
    previous: np.ndarray,
    frame_mask: np.ndarray,
    *,
    force_last: bool = False,
) -> Result:
    """Forward reduction for one decoded word.

    ``r[j]`` is the mass arriving at the decision at frame ``j``.  It includes
    probability entering from an exact previous boundary at j.  The recurrence
    is deliberately the project's own implementation; paths are not enumerated.
    """
    n = p.shape[0]
    p_eff = np.zeros(n, dtype=np.float64)
    p_eff[frame_mask] = p[frame_mask]
    forced_index = n - 1 if force_last and n else -1
    if forced_index >= 0:
        p_eff[forced_index] = 1.0

    r = np.zeros(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    if n == 0:
        return Result(r=r, alpha=alpha, p_eff=p_eff, wait=0.0, forced_index=-1)

    r[0] = previous[0]
    for j in range(1, n):
        r[j] = previous[j] + (1.0 - p_eff[j - 1]) * r[j - 1]
    alpha = constraint * p_eff * r
    wait = (1.0 - p_eff[-1]) * r[-1]
    return Result(
        r=r,
        alpha=alpha,
        p_eff=p_eff,
        wait=float(wait),
        forced_index=forced_index,
    )


def row_vjp(
    grad_alpha: np.ndarray,
    workspace: Result,
    constraint: np.ndarray,
    *,
    wait_upstream: float = 0.0,
) -> Result:
    """Vector-Jacobian product for :func:`row_forward`."""
    r = workspace.r
    p_eff = workspace.p_eff
    n = r.shape[0]
    grad_p_eff = np.zeros(n, dtype=np.float64)
    grad_previous = np.zeros(n, dtype=np.float64)
    if n == 0:
        return Result(grad_p_eff=grad_p_eff, grad_previous=grad_previous)

    # wait = (1 - p_last) * r_last
    grad_r = float(wait_upstream) * (1.0 - p_eff[-1])
    grad_p_eff[-1] -= float(wait_upstream) * r[-1]

    for j in range(n - 1, -1, -1):
        cj = constraint[j]
        grad_r += grad_alpha[j] * cj * p_eff[j]
        grad_p_eff[j] += grad_alpha[j] * cj * r[j]
        grad_previous[j] = grad_r
        if j > 0:
            # r[j] = previous[j] + (1-p[j-1])*r[j-1]
            grad_p_eff[j - 1] -= grad_r * r[j - 1]
            grad_r = grad_r * (1.0 - p_eff[j - 1])
    return Result(grad_p_eff=grad_p_eff, grad_previous=grad_previous)


def boundary_distribution(
    e: np.ndarray | None = None,
    constraints: Any | None = None,
    *,
    selection_scores: np.ndarray | None = None,
    p: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    stop: int | None = None,
    valid: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
    received_stop: int | None = None,
    start_boundary: int = 0,
    force_end: bool = False,
    last_valid_frame: int | None = None,
    initial_mass: float = 1.0,
    word_to_index: dict[str, int] | None = None,
    s: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    selection_energy: np.ndarray | None = None,
    **_ignored: Any,
) -> Result:
    """Compute constrained boundary probabilities and residual waiting mass.

    The returned ``alpha[i, j]`` is the mass of all monotone paths which end
    word ``i`` at j after applying human window factors.  ``wait[i]`` is the
    residual model mass still scanning for word i at the current block end.
    """
    if e is None:
        e = selection_scores if selection_scores is not None else selection_energy
    if e is None:
        e = s if s is not None else scores
    p_override = p if p is not None else probabilities
    if e is None and p_override is not None:
        p_override_arr = np.asarray(p_override, dtype=np.float64)
        # The reduction only needs p; use finite dummy logits for shape/results.
        e = np.zeros_like(p_override_arr, dtype=np.float64)
    if e is None:
        raise ValueError("boundary_distribution requires e/selection scores or p/probabilities")
    e2, squeezed = _as_2d(e)
    w, t_full = e2.shape
    n = t_full if stop is None else min(t_full, int(stop))

    base_mask = compose_frame_mask(
        n,
        valid=None if valid is None else np.asarray(valid)[:n],
        frame_mask=None if frame_mask is None else np.asarray(frame_mask)[:n],
        received_stop=received_stop,
        stop=n if frame_mask is None and received_stop is None else None,
    )

    effective_force = bool(force_end)
    force_index = -1
    if effective_force:
        if last_valid_frame is not None:
            force_index = int(last_valid_frame)
        elif valid is not None:
            valid_positions = np.flatnonzero(np.asarray(valid[:n], dtype=bool))
            force_index = int(valid_positions[-1]) if valid_positions.size else -1
        else:
            valid_positions = np.flatnonzero(base_mask)
            force_index = int(valid_positions[-1]) if valid_positions.size else n - 1
        if force_index >= n:
            visible_valid = np.flatnonzero(base_mask)
            force_index = int(visible_valid[-1]) if visible_valid.size else -1
        if force_index < 0 or force_index >= n:
            # No valid acoustic frame: there is intentionally no context/Z.
            n = 0
            effective_force = False
        else:
            n = force_index + 1
            base_mask = base_mask[:n]
            if valid is not None:
                base_mask &= np.asarray(valid[:n], dtype=bool)

    e_use = e2[:, :n]
    p = sigmoid(e_use) if n else np.zeros((w, 0))
    if p_override is not None and n:
        p_ov = np.asarray(p_override, dtype=np.float64)
        if p_ov.ndim == 1 and w == 1:
            p_ov = p_ov[None, :]
        p = p_ov[:, :n]
    c = _constraint_array(constraints, w, t_full, word_to_index)[:, :n]

    alpha = np.zeros((w, n), dtype=np.float64)
    waits = np.zeros(w, dtype=np.float64)
    previous = np.zeros(n, dtype=np.float64)
    if n:
        b = int(start_boundary)
        if 0 <= b < n:
            previous[b] = float(initial_mass)
        elif b == 0:
            previous[0] = float(initial_mass)

    for i in range(w):
        ws = row_forward(
            p[i],
            c[i],
            previous,
            base_mask,
            force_last=effective_force,
        )
        alpha[i] = ws.alpha
        waits[i] = ws.wait
        previous = ws.alpha

    total_mass = float(alpha[-1].sum()) if w and n else 0.0
    row_mass = alpha.sum(axis=1) if w else np.zeros(0)
    cumulative_wait = 1.0 - row_mass
    if n == 0 and w:
        # Before the first observable frame, the initial path mass is waiting.
        waits[0] = float(initial_mass)
    out = Result(
        alpha=alpha[0] if squeezed else alpha,
        boundary_probs=alpha[0] if squeezed else alpha,
        boundary_probability=alpha[0] if squeezed else alpha,
        wait=float(waits[0]) if squeezed else waits,
        stage_wait=float(waits[0]) if squeezed else waits,
        stage_waiting_mass=float(waits[0]) if squeezed else waits,
        cumulative_wait=float(cumulative_wait[0]) if squeezed else cumulative_wait,
        row_unselected_mass=float(cumulative_wait[0]) if squeezed else cumulative_wait,
        wait_prob=float(waits[0]) if squeezed else waits,
        waiting_probability=float(waits[0]) if squeezed else waits,
        waiting_mass=float(waits[0]) if squeezed else waits,
        total_waiting_mass=float(waits.sum()),
        incomplete_mass=float(max(0.0, 1.0 - total_mass)),
        p=p[0] if squeezed else p,
        e=e_use[0] if squeezed else e_use,
        constraints=c[0] if squeezed else c,
        frame_mask=base_mask,
        force_end=effective_force,
        forced_index=n - 1 if effective_force else -1,
        total_mass=total_mass,
        path_mass=total_mass,
        n_frames=n,
        start_boundary=int(start_boundary),
    )
    out["squeezed"] = squeezed
    return out


def boundary_forward(*args: Any, **kwargs: Any) -> Result:
    """Friendly alias for :func:`boundary_distribution`."""
    return boundary_distribution(*args, **kwargs)


def boundary_vjp(
    e: np.ndarray,
    forward_result: Result,
    grad_alpha: np.ndarray,
    constraints: Any | None = None,
    *,
    wait_upstream: np.ndarray | float | None = None,
    valid: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
    received_stop: int | None = None,
    start_boundary: int = 0,
    word_to_index: dict[str, int] | None = None,
) -> Result:
    """Backpropagate through :func:`boundary_distribution`.

    The public force-end frame and non-acoustic/padding positions are treated
    as constants (p=1 and p=0 respectively), so they receive no energy grad.
    """
    e2, squeezed_e = _as_2d(e)
    alpha_full = np.atleast_2d(np.asarray(forward_result.alpha, dtype=np.float64))
    w, n = alpha_full.shape
    grad_alpha_full = np.atleast_2d(np.asarray(grad_alpha, dtype=np.float64))
    if grad_alpha_full.shape != alpha_full.shape:
        raise ValueError("grad_alpha shape does not match alpha")

    base_mask = compose_frame_mask(
        n,
        valid=None if valid is None else np.asarray(valid)[:n],
        frame_mask=None if frame_mask is None else np.asarray(frame_mask)[:n],
        received_stop=received_stop,
    )
    if forward_result.get("force_end", False):
        base_mask[n - 1] = True
    c = _constraint_array(constraints, w, max(e2.shape[1], n), word_to_index)[:, :n]
    p = sigmoid(e2[:, :n]) if n else np.zeros((w, 0))

    grad_e = np.zeros((w, e2.shape[1]), dtype=np.float64)
    g_next = np.zeros(n, dtype=np.float64)
    wait_up = np.zeros(w, dtype=np.float64)
    if wait_upstream is not None:
        wu = np.asarray(wait_upstream, dtype=np.float64)
        if wu.ndim == 0:
            wait_up[:] = float(wu)
        else:
            wait_up = wu.reshape(w)

    for i in range(w - 1, -1, -1):
        if i == 0:
            previous = np.zeros(n, dtype=np.float64)
            if n and 0 <= int(start_boundary) < n:
                previous[int(start_boundary)] = 1.0
        else:
            previous = alpha_full[i - 1]
        ws = row_forward(
            p[i],
            c[i],
            previous,
            base_mask,
            force_last=bool(forward_result.get("force_end", False)),
        )
        ga = grad_alpha_full[i] + g_next
        rb = row_vjp(ga, ws, c[i], wait_upstream=float(wait_up[i]))
        differentiable = base_mask.copy()
        if ws.forced_index >= 0:
            differentiable[ws.forced_index] = False
        ge_row = np.zeros(n, dtype=np.float64)
        ge_row[differentiable] = (
            rb.grad_p_eff[differentiable] * p[i, differentiable] * (1.0 - p[i, differentiable])
        )
        grad_e[i, :n] = ge_row
        g_next = rb.grad_previous

    if squeezed_e:
        return Result(grad_e=grad_e[0], grad_initial_mass=float(g_next[0]) if n else 0.0)
    return Result(grad_e=grad_e, grad_initial_mass=float(g_next[0]) if n else 0.0)
