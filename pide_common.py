"""
Numerical utilities shared by every Hawkes-PIDE solver in this project
(risk-neutral and risk-averse alike). Nothing here depends on which utility
function is being solved for -- it only depends on the (lambda, h) grid
topology, which is the same across both HJB-PIDEs.
"""

import os
import time

import numpy as np
import scipy.sparse as sp


def build_jacobian_sparsity(N, M, n_tilde):
    """Sparsity pattern of the Jacobian of a Hawkes-PIDE right-hand side on an
    (N, M) = (lambda, h) grid: the equation at node (n,m) only depends on
    itself, its 2 neighbors in lambda, its 2 neighbors in h, and the node
    shifted by the Hawkes jump (n+n_tilde, clipped to N-1). This is <0.05%
    dense for the grids typically used here -- passing it to solve_ivp
    prevents Radau from estimating a DENSE Jacobian by finite differences
    (expensive both in memory and in RHS evaluations). The pattern is the
    same regardless of which utility (risk-neutral or CARA) sits in the RHS,
    since it only reflects which grid neighbors each node's finite-difference
    stencil touches.
    """
    size = N * M

    def idx(n, m):
        return n * M + m

    rows, cols = [], []
    for n in range(N):
        n_shift = min(n + n_tilde, N - 1)
        for m in range(M):
            i = idx(n, m)
            rows.append(i); cols.append(i)
            if n > 0:
                rows.append(i); cols.append(idx(n - 1, m))
            if n < N - 1:
                rows.append(i); cols.append(idx(n + 1, m))
            if m > 0:
                rows.append(i); cols.append(idx(n, m - 1))
            if m < M - 1:
                rows.append(i); cols.append(idx(n, m + 1))
            rows.append(i); cols.append(idx(n_shift, m))

    data = np.ones(len(rows))
    return sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def _get_rss_gb():
    """Resident memory (RSS) of the current process, in GB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1e6
        except Exception:
            pass
        return float("nan")


def make_logging_rhs(rhs_func, t_0, T_final, log_every_seconds=30.0, label=""):
    """Wraps rhs_func(t, y, *args) to print progress every log_every_seconds
    wall-clock seconds, without altering the original logic."""
    state = {"n_calls": 0, "t_start_wall": None, "last_log_wall": None}

    def wrapped(t, y, *args, **kwargs):
        state["n_calls"] += 1
        now = time.time()
        if state["t_start_wall"] is None:
            state["t_start_wall"] = now
            state["last_log_wall"] = now
        if (now - state["last_log_wall"]) >= log_every_seconds:
            state["last_log_wall"] = now
            elapsed = now - state["t_start_wall"]
            span = T_final - t_0
            pct = 100.0 * (T_final - t) / span if span > 0 else float("nan")
            prefix = f"[{label}] " if label else ""
            print(
                f"{prefix}t={t:.5f}  ~{pct:5.1f}% complete  "
                f"calls={state['n_calls']:6d}  "
                f"elapsed={elapsed:7.1f}s  "
                f"RSS={_get_rss_gb():5.2f} GB"
            )
        return rhs_func(t, y, *args, **kwargs)

    return wrapped
