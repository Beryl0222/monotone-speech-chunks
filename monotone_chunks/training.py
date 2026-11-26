"""Training objective, explicit denominators, and analytic gradients."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .boundary import boundary_distribution, boundary_vjp
from .constraints import geometric_feasibility
from .result import Result
from .sentence import SentenceInputs
from .windows import context_forward_backward


def mse_context_loss(context: np.ndarray, target: np.ndarray, normalize: bool = True) -> Result:
    """Mean/sum squared-error caller loss and its explicit upstream gradient."""
    ctx = np.asarray(context, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    diff = ctx - tgt
    denom = float(diff.size) if normalize else 1.0
    loss = float(np.sum(diff * diff) / denom)
    upstream = 2.0 * diff / denom
    return Result(loss=loss, context_upstream=upstream, denominator=denom)


def assess_sentence(
    sentence: SentenceInputs,
    *,
    arrived_stop: int | None = None,
    confirmed_stop: int | None = None,
    sealed: bool = False,
    force_end: bool | None = None,
    complete: bool | None = None,
    start_boundary: int = 0,
    min_path_mass: float = 0.0,
) -> Result:
    """Classify coverage, geometric conflicts, and soft allowed-path mass."""
    if isinstance(sentence, Mapping):
        sentence = SentenceInputs.from_materials(sentence)
    stop = sentence.num_frames if arrived_stop is None else min(sentence.num_frames, int(arrived_stop))
    do_force = bool(sealed if force_end is None else force_end)
    c = None
    if sentence.constraints is not None:
        c = sentence.constraints.factors_matrix(stop, sentence.num_words)
    geo = geometric_feasibility(
        c,
        stop,
        valid=sentence.valid,
        start_boundary=start_boundary,
        num_words=sentence.num_words,
    )

    max_hi = sentence.constraints.max_hi() if sentence.constraints is not None else None
    required_stop = sentence.last_valid_frame(stop)
    required_stop = 0 if required_stop is None else required_stop + 1
    if max_hi is not None:
        required_stop = max(required_stop, max_hi + 1)
    coverage_complete = bool(
        complete
        if complete is not None
        else (do_force or (arrived_stop is None and stop >= required_stop and required_stop > 0))
    )

    if not geo.feasible:
        # If the currently visible prefix is infeasible but the full tensor has
        # a future legal path, callers should wait for material rather than be
        # told that annotations conflict.
        full_c = None
        if sentence.constraints is not None:
            full_c = sentence.constraints.factors_matrix(sentence.num_frames, sentence.num_words)
        full_geo = geometric_feasibility(
            full_c,
            sentence.num_frames,
            valid=sentence.valid,
            start_boundary=start_boundary,
            num_words=sentence.num_words,
        )
        if full_geo.feasible and not coverage_complete:
            status = "materials_incomplete"
            eligible = False
        else:
            status = "constraints_infeasible"
            eligible = False
        reason = geo.reason
    elif not coverage_complete or stop < required_stop:
        status = "materials_incomplete"
        reason = "constraint_hi_or_end_not_covered"
        eligible = False
    else:
        status = "feasible"
        reason = "ok"
        eligible = True

    # Soft mass is computed only when geometry is possible.  It distinguishes a
    # merely tiny set of probable allowed paths from a hard conflict.
    path_mass = 0.0
    if geo.feasible and stop > 0:
        e = sentence.selection_energies(stop)
        dist = boundary_distribution(
            e,
            c,
            valid=sentence.valid[:stop],
            force_end=do_force,
            start_boundary=start_boundary,
        )
        path_mass = dist.path_mass
        if eligible and path_mass <= float(min_path_mass):
            status = "insufficient_allowed_path_mass"
            reason = "constrained_soft_path_mass_below_minimum"
            eligible = False

    state_aliases = {
        "feasible": "ready",
        "materials_incomplete": "waiting_for_materials",
        "constraints_infeasible": "conflict",
        "insufficient_allowed_path_mass": "allowed_paths_insufficient",
    }
    return Result(
        eligible=eligible,
        status=status,
        state=state_aliases.get(status, status),
        classification=status,
        conflict=status == "constraints_infeasible",
        materials_incomplete=status == "materials_incomplete",
        constraints_infeasible=status == "constraints_infeasible",
        allowed_paths_insufficient=status == "insufficient_allowed_path_mass",
        reason=reason,
        coverage_complete=coverage_complete,
        arrived_stop=stop,
        required_stop=required_stop,
        geometric=geo,
        allowed_path_count=geo.allowed_path_count,
        path_mass=float(path_mass),
        path_mass_denominator=float(path_mass),
        confirmed_stop=confirmed_stop,
        sealed=sealed,
    )


def train_forward_backward(
    e: np.ndarray | None = None,
    u: np.ndarray | None = None,
    V: np.ndarray | None = None,
    chunk_width: int | None = None,
    *,
    Q: np.ndarray | None = None,
    Ks: np.ndarray | None = None,
    Kc: np.ndarray | None = None,
    b_select: np.ndarray | float = 0.0,
    scale: float | None = None,
    constraints: Any | None = None,
    word_to_index: dict[str, int] | None = None,
    context_upstream: np.ndarray | None = None,
    context_loss_value: float | None = None,
    context_target: np.ndarray | None = None,
    context_loss_weight: float = 1.0,
    nll_weight: float = 1.0,
    context_normalize: bool = True,
    chunk_size: int | None = None,
    width: int | None = None,
    stop: int | None = None,
    valid: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
    received_stop: int | None = None,
    start_boundary: int = 0,
    force_end: bool = False,
    last_valid_frame: int | None = None,
    wait_upstream: np.ndarray | float | None = None,
    selection_energy: np.ndarray | None = None,
    chunk_energy: np.ndarray | None = None,
    selection_scores: np.ndarray | None = None,
    chunk_scores: np.ndarray | None = None,
) -> Result:
    """Run soft forward, context construction, NLL, and analytic VJPs.

    ``context_upstream`` is the caller-provided adjoint of context.  If no
    explicit ``context_loss_value`` is given, the returned context term is the
    linear surrogate ``sum(context_upstream * context)`` so finite differences
    can check every returned gradient.
    """
    if chunk_width is None:
        chunk_width = chunk_size if chunk_size is not None else width
    if chunk_width is None:
        raise ValueError("chunk_width (or chunk_size) is required")
    e = e if e is not None else selection_energy
    e = e if e is not None else selection_scores
    u = u if u is not None else chunk_energy
    u = u if u is not None else chunk_scores

    if e is None:
        if Q is None or Ks is None:
            raise ValueError("either e/selection energy or Q and Ks are required")
        q = np.asarray(Q, dtype=np.float64)
        ks = np.asarray(Ks, dtype=np.float64)
        w = q.shape[0]
        n_full = ks.shape[0] if stop is None else min(ks.shape[0], int(stop))
        bias = np.asarray(b_select, dtype=np.float64)
        if bias.ndim == 0:
            bias_part = float(bias)
        else:
            bias_part = bias[:w, None] if bias.shape[0] == w else 0.0
        e = q @ ks[:n_full].T + bias_part
    else:
        e = np.asarray(e, dtype=np.float64)
        if e.ndim == 1:
            e = e[None, :]

    w, e_input_width = e.shape
    t_e = e_input_width
    n = t_e if stop is None else min(t_e, int(stop))
    e = e[:, :n]

    u_input_width: int | None = None
    if u is None:
        if Q is None or Kc is None:
            raise ValueError("either u/chunk energy or Q and Kc are required")
        q = np.asarray(Q, dtype=np.float64)
        kc = np.asarray(Kc, dtype=np.float64)
        d = q.shape[1]
        s = 1.0 / np.sqrt(d) if scale is None else float(scale)
        u = (q @ kc[:n].T) * s
    else:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            u = u[None, :]
        u_input_width = int(u.shape[1])
        u = u[:, :n]
        if scale is None and Q is not None:
            scale = 1.0 / np.sqrt(np.asarray(Q).shape[1])
        elif scale is None:
            scale = 1.0
    if u_input_width is None:
        u_input_width = int(u.shape[1])
    if V is None:
        raise ValueError("V is required")
    V_full = np.asarray(V, dtype=np.float64)
    v = V_full[:n]
    if chunk_width is None:
        raise ValueError("chunk_width is required")

    dist = boundary_distribution(
        e,
        constraints,
        stop=n,
        valid=valid,
        frame_mask=frame_mask,
        received_stop=received_stop,
        start_boundary=start_boundary,
        force_end=force_end,
        last_valid_frame=last_valid_frame,
        word_to_index=word_to_index,
    )
    alpha = dist.alpha
    if alpha.ndim == 1:
        alpha = alpha[None, :]
    # Force-end/no-valid-frame handling can shorten the active range after the
    # boundary reduction identifies the true terminal frame.
    n = int(dist.n_frames)
    e = e[:, :n]
    u = u[:, :n]
    v = v[:n]
    wait = np.atleast_1d(dist.wait)
    stage_wait = np.atleast_1d(dist.stage_wait)
    cumulative_wait = np.atleast_1d(dist.cumulative_wait)
    z = float(dist.path_mass)

    value_dim = v.shape[1]
    context = np.zeros((w, value_dim), dtype=np.float64)
    grad_alpha = np.zeros_like(alpha)
    grad_u = np.zeros_like(u)
    grad_v = np.zeros_like(v)

    gctx_all: np.ndarray | None
    if context_upstream is not None:
        gctx_all = np.asarray(context_upstream, dtype=np.float64)
        if gctx_all.shape != (w, value_dim):
            # Allow a single shared upstream vector.
            if gctx_all.shape == (value_dim,):
                gctx_all = np.broadcast_to(gctx_all, (w, value_dim))
            else:
                raise ValueError(f"context_upstream must have shape {(w, value_dim)}")
    else:
        gctx_all = None

    def run_context_backward(upstream_all: np.ndarray | None) -> None:
        nonlocal grad_v
        for i in range(w):
            cb = context_forward_backward(
                u[i],
                v,
                int(chunk_width),
                alpha=alpha[i],
                context_upstream=None if upstream_all is None else upstream_all[i],
                frame_mask=dist.frame_mask,
            )
            context[i] = cb.context
            if upstream_all is not None:
                grad_alpha[i] = cb.grad_alpha
                grad_u[i] = cb.grad_u
                grad_v += cb.grad_v

    # First pass always materializes contexts without accumulating adjoints.
    run_context_backward(None)
    mse_info = None
    if context_target is not None:
        target = np.asarray(context_target, dtype=np.float64)
        mse_info = mse_context_loss(context, target, normalize=context_normalize)
        gctx_all = mse_info.context_upstream * float(context_loss_weight)
        run_context_backward(gctx_all)
    elif gctx_all is not None:
        run_context_backward(gctx_all)

    if z > 0.0:
        nll = float(-float(nll_weight) * np.log(z))
    else:
        nll = float("inf") if float(nll_weight) != 0.0 else 0.0
    if context_loss_value is not None:
        context_loss = float(context_loss_value)
    elif mse_info is not None:
        context_loss = float(context_loss_weight) * float(mse_info.loss)
    elif gctx_all is not None and context_upstream is not None:
        context_loss = float(np.sum(np.asarray(context_upstream) * context))
    else:
        context_loss = 0.0
    loss = nll + context_loss if np.isfinite(nll) else float("inf")

    if z > 0.0 and float(nll_weight) != 0.0:
        grad_alpha[-1] -= float(nll_weight) / z
    boundary_grads = boundary_vjp(
        e,
        dist,
        grad_alpha,
        constraints,
        wait_upstream=wait_upstream,
        valid=valid,
        frame_mask=frame_mask,
        received_stop=received_stop,
        start_boundary=start_boundary,
        word_to_index=word_to_index,
    )
    grad_e_active = boundary_grads.grad_e
    grad_u_active = grad_u

    # Gradients wrt caller-provided score matrices retain their original width;
    # unused/padding/terminal-overridden columns are exact zeros.
    grad_e_full = np.zeros((w, e_input_width), dtype=np.float64)
    grad_e_full[:, :n] = grad_e_active
    u_width = int(u_input_width if u_input_width is not None else n)
    grad_u_full = np.zeros((w, u_width), dtype=np.float64)
    grad_u_full[:, :n] = grad_u_active

    grads = Result(
        grad_e=grad_e_full,
        grad_selection_energy=grad_e_full,
        grad_selection_scores=grad_e_full,
        grad_selection=grad_e_full,
        grad_monotone_scores=grad_e_full,
        grad_s=grad_e_full,
        grad_u=grad_u_full,
        grad_chunk_energy=grad_u_full,
        grad_chunk_scores=grad_u_full,
        grad_chunk=grad_u_full,
        grad_in_chunk_scores=grad_u_full,
        grad_attention_scores=grad_u_full,
        grad_a=grad_u_full,
        grad_V=grad_v,
        grad_values=grad_v,
        grad_value=grad_v,
    )

    if Q is not None and Ks is not None:
        q = np.asarray(Q, dtype=np.float64)
        ks = np.asarray(Ks, dtype=np.float64)[:n]
        grads.grad_Q_selection = grad_e_active @ ks
        grads.grad_Ks = grad_e_active.T @ q
        bias = np.asarray(b_select, dtype=np.float64)
        if bias.ndim == 0:
            grads.grad_b_select = float(grad_e_active.sum())
        elif bias.shape[0] == w:
            grads.grad_b_select = grad_e_active.sum(axis=1)
        else:
            grads.grad_b_select = grad_e_active.sum()
    if Q is not None and Kc is not None:
        q = np.asarray(Q, dtype=np.float64)
        kc = np.asarray(Kc, dtype=np.float64)[:n]
        s = 1.0 / np.sqrt(q.shape[1]) if scale is None else float(scale)
        gu_scaled = grad_u_active * s
        grads.grad_Q_chunk = gu_scaled @ kc
        grads.grad_Kc = gu_scaled.T @ q
        grads.grad_K_chunk = grads.grad_Kc
        grads.grad_key_chunk = grads.grad_Kc
    if "grad_Ks" in grads:
        grads.grad_K_selection = grads.grad_Ks
        grads.grad_key_selection = grads.grad_Ks
    if "grad_Ks" in grads or "grad_Kc" in grads:
        grads.grad_keys = Result(
            Ks=grads.get("grad_Ks"),
            Kc=grads.get("grad_Kc"),
        )
    if "grad_Q_selection" in grads or "grad_Q_chunk" in grads:
        gq = np.zeros(np.asarray(Q).shape, dtype=np.float64)
        if "grad_Q_selection" in grads:
            gq += grads.grad_Q_selection
        if "grad_Q_chunk" in grads:
            gq += grads.grad_Q_chunk
        grads.grad_Q = gq
    if grad_v.shape != V_full.shape:
        gv_full = np.zeros_like(V_full, dtype=np.float64)
        gv_full[:n] = grad_v
        grads.grad_V = gv_full
        grads.grad_values = gv_full
        grads.grad_value = gv_full
    else:
        grads.grad_V = grad_v
        grads.grad_values = grad_v
        grads.grad_value = grad_v

    return Result(
        loss=loss,
        nll=nll,
        path_mass=z,
        path_mass_denominator=z,
        denominator=z,
        context_loss=context_loss,
        context_loss_denominator=None if mse_info is None else mse_info.denominator,
        context_upstream=gctx_all,
        context=context,
        alpha=alpha,
        boundary_probability=alpha,
        boundary_probs=alpha,
        wait=wait,
        wait_prob=wait,
        stage_wait=stage_wait,
        cumulative_wait=cumulative_wait,
        waiting_probability=wait,
        waiting_mass=wait,
        total_waiting_mass=float(np.sum(stage_wait)),
        e=e,
        u=u,
        V=v,
        frame_mask=dist.frame_mask,
        eligibility=Result(eligible=z > 0.0 and np.isfinite(nll), status="computed"),
        gradients=grads,
        **grads,
    )


# Friendly functional aliases.
monotone_chunk_forward = train_forward_backward
forward_backward = train_forward_backward
training_forward = train_forward_backward
training_step = train_forward_backward


def train_sentence(
    sentence: SentenceInputs,
    *,
    context_upstream: np.ndarray | None = None,
    arrived_stop: int | None = None,
    confirmed_stop: int | None = None,
    sealed: bool = False,
    force_end: bool | None = None,
    complete: bool | None = None,
    assess: bool = True,
    run_if_ineligible: bool = False,
    start_boundary: int = 0,
    wait_upstream: np.ndarray | None = None,
    **train_kwargs: Any,
) -> Result:
    """Assess one material sentence and, when eligible, run train fwd/bwd."""
    if isinstance(sentence, Mapping):
        sentence = SentenceInputs.from_materials(sentence)
    eligibility = assess_sentence(
        sentence,
        arrived_stop=arrived_stop,
        confirmed_stop=confirmed_stop,
        sealed=sealed,
        force_end=force_end,
        complete=complete,
        start_boundary=start_boundary,
    )
    if assess and not eligibility.eligible and not run_if_ineligible:
        return Result(
            eligible=False,
            eligibility=eligibility,
            status=eligibility.status,
            reason=eligibility.reason,
            sentence_id=sentence.utterance_id,
            sample_weight=sentence.sample_weight,
            loss=None,
            gradients=None,
        )

    g_up = context_upstream
    if g_up is None and sentence.context_upstream is not None:
        g_up = sentence.context_upstream
    do_force = bool(sealed if force_end is None else force_end)
    result = train_forward_backward(
        V=sentence.V,
        chunk_width=sentence.chunk_width,
        Q=sentence.Q,
        Ks=sentence.Ks,
        Kc=sentence.Kc,
        b_select=sentence.b_select,
        constraints=sentence.constraints,
        word_to_index={w: i for i, w in enumerate(sentence.word_ids)},
        context_upstream=g_up,
        stop=arrived_stop,
        valid=sentence.valid,
        start_boundary=start_boundary,
        force_end=do_force,
        last_valid_frame=sentence.last_valid_frame(arrived_stop),
        wait_upstream=wait_upstream,
        **train_kwargs,
    )
    result.update(
        eligible=bool(eligibility.eligible),
        eligibility=eligibility,
        status=eligibility.status,
        reason=eligibility.reason,
        sentence_id=sentence.utterance_id,
        sample_weight=sentence.sample_weight,
    )
    return result


def train_batch(
    sentences: Sequence[SentenceInputs],
    *,
    context_upstreams: Sequence[np.ndarray | None] | None = None,
    arrived_stops: Sequence[int | None] | None = None,
    sealed_flags: Sequence[bool] | None = None,
) -> Result:
    """Aggregate only fully covered, feasible sentences by sentence weight."""
    reports: list[Result] = []
    eligible: list[Result] = []
    denominator = 0.0
    for idx, sentence in enumerate(sentences):
        if isinstance(sentence, Mapping):
            sentence = SentenceInputs.from_materials(sentence)
        g = None if context_upstreams is None else context_upstreams[idx]
        stop = None if arrived_stops is None else arrived_stops[idx]
        sealed = False if sealed_flags is None else sealed_flags[idx]
        rep = train_sentence(
            sentence,
            context_upstream=g,
            arrived_stop=stop,
            sealed=sealed,
        )
        reports.append(rep)
        if rep.eligible:
            denominator += float(sentence.sample_weight)
            eligible.append(rep)

    total_loss = 0.0
    if denominator > 0:
        for sentence, rep in zip(sentences, reports):
            if rep.eligible:
                total_loss += float(sentence.sample_weight) * rep.loss / denominator

    return Result(
        loss=total_loss,
        eligible_weight_denominator=denominator,
        sentence_weight_denominator=denominator,
        denominator=denominator,
        num_sentences=len(sentences),
        num_eligible=len(eligible),
        reports=reports,
        eligible_reports=eligible,
    )
