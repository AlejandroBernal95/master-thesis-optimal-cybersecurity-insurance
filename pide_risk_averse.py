"""
Risk-averse (CARA utility) HJB-PIDE for the joint cybersecurity-investment /
cyberinsurance problem under Hawkes attack-arrival dynamics.

V(t,lambda,h) = -exp(-omega*(x + W(t,lambda,h))), certainty-equivalent ansatz.
W solves a semilinear PIDE where the jump term is driven by the MGF of the
insurance-retained loss (deductible D, coinsurance C), since CARA utility is
sensitive to the full jump distribution, not just its mean (unlike the
risk-neutral case in pide_risk_neutral.py).

Also includes the M(t,h) upper bound (deterministic control problem with the
loss terms dropped -- valid because they enter W non-positively), the
Q(t,lambda,h;C) lower-bound PIDE (eq. 4.29-4.30), and the resulting
W_min(t,lambda,h;C) lower bound (suboptimal policy z=rho*h plus the loss
penalty carried by Q), used both as the thesis-level bounds on W and as a
numerical safeguard (clipping) inside the PIDE's jump term.
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.sparse import lil_matrix

from premiums import GLI, calculate_premium_lomax, calculate_premium_lognormal
from pide_common import build_jacobian_sparsity, make_logging_rhs


# =============================================================================
# M(t,h): upper bound (deterministic problem, no losses)
# =============================================================================

def solve_M_upper_bound(
    t_0=0.0, T_final=1.0, K_steps=200,
    h_min=0.0, h_max=50.0, d_h=0.1,
    rho=0.2, gamma=0.05, delta=1.0,
):
    """
    M(t,h) = sup_z [ -int_t^T (delta*z + gamma/2 z^2) dv + sqrt(H_T) ]
    subject to dH = (z - rho*H) dt, H_t = h.
    Returns M_tensor (N_h, K_steps) and (h_grid, t_eval), with t_eval[0]=T,
    t_eval[-1]=0.
    """
    h_grid = np.arange(h_min, h_max + d_h / 2, d_h)
    M_T = np.sqrt(h_grid)

    t_span = (T_final, t_0)
    t_eval = np.linspace(T_final, t_0, K_steps)

    def rhs(t, M_flat):
        M = M_flat
        dM_dh = np.zeros_like(M)
        dM_dh[1:-1] = (M[2:] - M[:-2]) / (2.0 * d_h)
        dM_dh[0] = (M[1] - M[0]) / d_h
        dM_dh[-1] = (M[-1] - M[-2]) / d_h
        control_term = (np.maximum(dM_dh - delta, 0.0) ** 2) / (2.0 * gamma)
        return rho * h_grid * dM_dh - control_term

    sol = solve_ivp(
        fun=rhs, t_span=t_span, y0=M_T, method='Radau',
        t_eval=t_eval, rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"solve_M_upper_bound did not converge: {sol.message}")

    return sol.y, (h_grid, t_eval)


def build_M_interpolator(M_tensor, h_grid, t_eval):
    """Returns M_interp(t) -> array over h_grid (linear interpolation in t;
    h_grid must exactly match the RA solver's h_grid). interp1d requires an
    increasing x, so t_eval (which comes decreasing, from T to 0) is reversed."""
    t_incr = t_eval[::-1]
    M_incr = M_tensor[:, ::-1]
    return interp1d(t_incr, M_incr, axis=1, kind='linear',
                     bounds_error=False, fill_value=(M_incr[:, 0], M_incr[:, -1]))


# =============================================================================
# Q(t,lambda,h;C): lower-bound PIDE (eq. 4.29-4.30)
#
#   dQ/dt + xi*(alpha-lambda)*dQ/dlambda
#         + lambda*( ell(h;C)*Q(t,lambda+beta,h;C) - Q(t,lambda,h;C) ) = 0,
#   Q(T,lambda,h;C) = 1,
#
# where ell(h;C) = (1-S(h,v)) + S(h,v)*mgf_val is the same quantity used in
# the main W PIDE (mgf_val doesn't depend on h, only S(h,v) does).
#
# No d/dh term: h enters only as a fixed parameter through ell_grid, so
# there is no coupling across h-columns at all -- each h in h_grid solves
# an independent 1D-in-lambda linear PIDE. This is what makes Q much
# cheaper to solve than the full (lambda,h)-coupled W.
# =============================================================================

def compute_lambda_log_derivative_terms(L, d_lambda):
    """
    Returns an (N, M) array equal, elementwise, to (dQ/dlambda) / Q -- i.e.
    the lambda-derivative term needed by the log-space PIDE below -- using
    only exp() of DIFFERENCES between neighboring lambda grid points:

        (dQ/dlambda)/Q  =  (exp(L[n+1]-L[n]) - exp(L[n-1]-L[n])) / (2*d_lambda)

    These differences stay numerically bounded (they only span one grid
    step, or n_tilde grid steps for the jump term below) even when L
    itself grows to O(1000) or more, which is exactly what lets this
    formulation avoid the overflow that a direct Q-space finite difference
    hits once Q ~ exp(L) exceeds float64 range (see module notes on the
    Q(t,lambda,h;C) PIDE).

    N == 1 (lambda_min == lambda_max) has no lambda-neighbor to difference
    against, so this term is identically 0 there, same convention as the
    other lambda-derivative helpers in this module.
    """
    N, M = L.shape
    term = np.zeros_like(L)
    if N == 1:
        return term
    diff_plus = L[2:, :] - L[1:-1, :]      # L[n+1] - L[n], for n = 1..N-2
    diff_minus = L[1:-1, :] - L[:-2, :]    # L[n] - L[n-1], for n = 1..N-2
    term[1:-1, :] = (np.exp(diff_plus) - np.exp(-diff_minus)) / (2.0 * d_lambda)
    term[0, :] = (np.exp(L[1, :] - L[0, :]) - 1.0) / d_lambda
    term[-1, :] = (1.0 - np.exp(L[-2, :] - L[-1, :])) / d_lambda
    return term


def PDE_rhs_logQ(t, L_flat, N, M, lambda_grid, d_lambda, n_tilde, xi, alpha, ell_grid):
    """
    RHS for the lower-bound PIDE, solved in log-space: L = ln(Q). Q itself
    is never materialized -- only exp() of DIFFERENCES of L between grid
    points close together (one lambda step, or n_tilde lambda steps for
    the jump term), which is what keeps this numerically stable even when
    ell(h;C) is large (e.g. C < 1 with omega not small: retained losses
    are no longer capped at the deductible D the way they are at C=1, so
    the CARA moment -- and with it ell(h;C) and Q's growth rate -- can be
    far larger than the C=1 case).

    Derivation: from dQ/dt = -xi*(alpha-lambda)*dQ/dlambda
                              - lambda*(ell(h;C)*Q_shift - Q),
    and Q = exp(L), dividing both sides by Q = exp(L) gives
        dL/dt = -xi*(alpha-lambda)*(dQ/dlambda)/Q
                 - lambda*(ell(h;C)*exp(L_shift - L) - 1),
    where (dQ/dlambda)/Q is computed via compute_lambda_log_derivative_terms.
    Terminal condition Q(T,.,.)=1 becomes L(T,.,.)=0.
    """
    L = L_flat.reshape((N, M))
    lambda_deriv_term = compute_lambda_log_derivative_terms(L, d_lambda)

    n_indices = np.arange(N)
    n_shift = np.minimum(n_indices + n_tilde, N - 1)
    diff_shift = L[n_shift, :] - L  # L_shift - L, bounded (spans n_tilde grid steps)

    jump_rate_term = lambda_grid[:, None] * (ell_grid[None, :] * np.exp(diff_shift) - 1.0)

    dL_dt = (
        -xi * (alpha - lambda_grid[:, None]) * lambda_deriv_term
        - jump_rate_term
    )
    return dL_dt.flatten()


def build_jacobian_sparsity_Q(N, M, n_tilde):
    """Sparsity pattern for the log-space Q Jacobian (same coupling
    structure as the direct-Q formulation this replaced: flattened index
    for (n, m) is n*M + m). No h-coupling at all: entries only within the
    same h-column (same m), linking n to n-1, n, n+1 (central difference
    stencil) and to n_shift = min(n+n_tilde, N-1) (jump term)."""
    size = N * M
    jac = lil_matrix((size, size), dtype=bool)
    for n in range(N):
        n_shift = min(n + n_tilde, N - 1)
        for m in range(M):
            row = n * M + m
            jac[row, row] = True
            if n > 0:
                jac[row, (n - 1) * M + m] = True
            if n < N - 1:
                jac[row, (n + 1) * M + m] = True
            jac[row, n_shift * M + m] = True
    return jac.tocsr()


def solve_logQ_lower_bound(
    t_0, T_final, K_steps,
    lambda_min, lambda_max, d_lambda,
    h_min, h_max, d_h,
    xi, alpha, n_tilde,
    ell_grid,
):
    """Solves L(t,lambda,h;C) = ln Q(t,lambda,h;C) on the given grid, in
    log-space throughout (see PDE_rhs_logQ). Terminal condition L(T,.,.)=0
    (i.e. Q(T,.,.)=1). Returns (logQ_tensor, (lambda_grid, h_grid, t_eval)).

    ell_grid must already be evaluated on h_grid (shape (M,)) and use the
    SAME C, omega, r, and distribution parameters as the main W solve --
    pass in the ell(h;C) built from the mgf_val already computed by
    MGF_Bounded_Losses_RA, don't recompute it here. n_tilde must match the
    one used for the main W solve (same beta/d_lambda discretization).
    """
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    N = len(lambda_grid)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    M = len(h_grid)

    if ell_grid.shape[0] != M:
        raise ValueError(
            f"ell_grid has shape {ell_grid.shape}, expected ({M},) to match h_grid. "
            "Make sure ell_grid was built on the same h_grid passed here."
        )

    L_T = np.zeros((N, M))  # ln(1) = 0
    t_span = (T_final, t_0)
    t_eval = np.linspace(T_final, t_0, K_steps)

    extra_args = (N, M, lambda_grid, d_lambda, n_tilde, xi, alpha, ell_grid)

    jac_sparsity = build_jacobian_sparsity_Q(N, M, n_tilde)

    sol = solve_ivp(
        fun=PDE_rhs_logQ, t_span=t_span, y0=L_T.flatten(), method='Radau',
        t_eval=t_eval, args=extra_args, rtol=1e-8, atol=1e-10,
        jac_sparsity=jac_sparsity,
    )

    if not sol.success:
        raise RuntimeError(
            f"solve_ivp did not converge in solve_logQ_lower_bound (status={sol.status}): "
            f"{sol.message}\nCompleted {sol.y.shape[1]} of {len(t_eval)} time points."
        )

    logQ_tensor = sol.y.reshape((N, M, len(t_eval)))
    return logQ_tensor, (lambda_grid, h_grid, t_eval)


def build_logQ_interpolator(logQ_tensor, lambda_grid, h_grid, t_eval):
    """Returns logQ_interp(t) -> (N, M) array of ln Q(t,lambda,h;C).
    t_eval comes decreasing (T_final -> t_0); RegularGridInterpolator
    needs strictly increasing coordinates, so t and the corresponding
    slices of logQ_tensor are reversed before building the interpolator.
    Interpolating in log-space (rather than interpolating Q and taking
    log afterward) avoids ever materializing Q, consistent with the rest
    of this section."""
    t_incr = t_eval[::-1]
    logQ_incr = logQ_tensor[:, :, ::-1]

    rgi = RegularGridInterpolator(
        (lambda_grid, h_grid, t_incr), logQ_incr,
        bounds_error=False, fill_value=None,
    )

    lam_mesh, h_mesh = np.meshgrid(lambda_grid, h_grid, indexing='ij')

    def logQ_interp(t):
        pts = np.stack([lam_mesh.ravel(), h_mesh.ravel(),
                         np.full(lam_mesh.size, t)], axis=1)
        return rgi(pts).reshape(lam_mesh.shape)

    return logQ_interp


# =============================================================================
# W_min(t,lambda,h;C): lower bound for W (eq. 4.24-4.26)
# =============================================================================

def W_min_closed_form(t, lambda_grid, h_grid, T_final, rho, gamma, delta, omega, logQ_interp):
    """
    W_min(t,lambda,h;C) = sqrt(h)
                           - rho*h*(delta + gamma*rho*h/2)*(T_final - t)
                           - (1/omega) * ln Q(t,lambda,h;C)

    The first two terms are the value of the loss-free, suboptimal policy
    z=rho*h; the log(Q) term carries the actual loss/risk information and
    the lambda-dependence. logQ_interp already returns ln Q directly (Q is
    solved for in log-space, see solve_logQ_lower_bound, to avoid overflow
    when ell(h;C) is large) so no log() call or clipping guard is needed
    here anymore. Returns an (N, M) array.
    """
    tau = T_final - t
    h_row = h_grid[None, :]
    base_h_only = np.sqrt(h_row) - rho * h_row * (delta + gamma * rho * h_row / 2.0) * tau
    base = np.tile(base_h_only, (len(lambda_grid), 1))  # (N, M)

    log_term = logQ_interp(t) / omega  # (N, M)

    return base - log_term


# =============================================================================
# MGF of the insurance-retained loss (deductible D, coinsurance C)
# =============================================================================

def _safe_exp_RA(val, max_exponent=700.0):
    return np.exp(np.clip(val, a_min=None, a_max=max_exponent))


def f_Bounded_eta_RA(dist_type, y, k=None, eta_0=None, mu=None, sigma=None):
    dist_type = dist_type.lower()
    if dist_type == 'lomax':
        return k * (eta_0 ** k) / ((y + eta_0) ** (k + 1))
    elif dist_type == 'lognormal':
        return (1.0 / (y * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(y) - mu) ** 2) / (2 * (sigma ** 2))
        )
    raise ValueError(f"Unsupported distribution '{dist_type}'.")


def term_1_MGF_BL_RA(dist_type, omega, D, k=None, eta_0=None, mu=None, sigma=None):
    if D == 0:
        return 0.0
    integrand = lambda y: _safe_exp_RA(omega * y) * f_Bounded_eta_RA(dist_type, y, k=k, eta_0=eta_0, mu=mu, sigma=sigma)
    v, _ = quad(integrand, 0, D, limit=100, epsabs=1e-8, epsrel=1e-8)
    return v


def term_2_MGF_BL_RA(dist_type, C, omega, r, D, k=None, eta_0=None, mu=None, sigma=None):
    if D >= r:
        return 0.0
    integrand = lambda y: _safe_exp_RA(omega * ((1 - C) * y + C * D)) * f_Bounded_eta_RA(dist_type, y, k=k, eta_0=eta_0, mu=mu, sigma=sigma)
    v, _ = quad(integrand, D, r, limit=100, epsabs=1e-8, epsrel=1e-8)
    return v


def term_3_MGF_BL_RA(dist_type, C, omega, r, D, k=None, eta_0=None, mu=None, sigma=None):
    if dist_type == 'lomax':
        surv = (eta_0 / (eta_0 + r)) ** k
    else:
        surv = norm.cdf((mu - np.log(r)) / sigma)
    return surv * _safe_exp_RA(omega * ((1 - C) * r + C * D))


def MGF_Bounded_Losses_RA(dist_type, C, omega=0.1, r=10000, k=None, eta_0=None,
                           mu=None, sigma=None, D=None, d=None, return_log=False,
                           return_minus_1=False):
    dist_type = dist_type.lower()
    if not (0 <= C <= 1):
        raise ValueError("C has to be in [0,1].")
    if (D is None) == (d is None):
        raise ValueError("Exactly one of D or d must be given.")
    if d is not None:
        exp_eta_i_mean = eta_0 / (k - 1) if dist_type == 'lomax' else np.exp(mu + (sigma ** 2) / 2.0)
        D = d * exp_eta_i_mean
    if not (0 <= D < r):
        raise ValueError(f"Deductible D={D:.4f} must satisfy 0<=D<r (r={r}).")

    if return_minus_1 and return_log:
        raise ValueError("return_minus_1 and return_log cannot both be True.")

    if C == 0:
        # With no coinsurance the retained loss equals the full loss for any
        # D (term_2's (1-C)*y + C*D reduces to y), so splitting the integral
        # at D is unnecessary -- integrate the whole (0, r) range in one piece.
        # D is still required above because Initial_Net_Value_Hawkes_RA / V_RA
        # reuse it downstream for the premium (same reasoning as the
        # risk-neutral solver's ghost-input notes).
        integrand = lambda y: _safe_exp_RA(omega * y) * f_Bounded_eta_RA(dist_type, y, k=k, eta_0=eta_0, mu=mu, sigma=sigma)
        t12, _ = quad(integrand, 0, r, limit=200, epsabs=1e-8, epsrel=1e-8)
        t3 = term_3_MGF_BL_RA(dist_type, C, omega, r, D, k=k, eta_0=eta_0, mu=mu, sigma=sigma)

        if return_minus_1:
            mgf_val = t12 + t3
            mgf_minus_1 = _mgf_minus_1_RA(dist_type, C, omega, r, D, k=k, eta_0=eta_0,
                                           mu=mu, sigma=sigma)
            return mgf_val, mgf_minus_1
        if return_log:
            terms = np.array([t12, t3])
            valid = terms[terms > 0]
            return -np.inf if len(valid) == 0 else logsumexp(np.log(valid))
        return t12 + t3

    t1 = term_1_MGF_BL_RA(dist_type, omega, D, k=k, eta_0=eta_0, mu=mu, sigma=sigma)
    t2 = term_2_MGF_BL_RA(dist_type, C, omega, r, D, k=k, eta_0=eta_0, mu=mu, sigma=sigma)
    t3 = term_3_MGF_BL_RA(dist_type, C, omega, r, D, k=k, eta_0=eta_0, mu=mu, sigma=sigma)

    if return_minus_1:
        mgf_val = t1 + t2 + t3
        mgf_minus_1 = _mgf_minus_1_RA(dist_type, C, omega, r, D, k=k, eta_0=eta_0,
                                       mu=mu, sigma=sigma)
        return mgf_val, mgf_minus_1
    if return_log:
        terms = np.array([t1, t2, t3])
        valid = terms[terms > 0]
        return -np.inf if len(valid) == 0 else logsumexp(np.log(valid))
    return t1 + t2 + t3


def _mgf_minus_1_RA(dist_type, C, omega, r, D, k=None, eta_0=None, mu=None, sigma=None):
    """
    Numerically stable computation of E[exp(omega*Y)] - 1, where
    Y = eta^r - C*(eta^r - D)^+ is the (truncated, coinsured) retained
    loss -- i.e. mgf_val - 1 from MGF_Bounded_Losses_RA, WITHOUT ever
    forming that subtraction in floating point (which loses precision
    catastrophically once omega is small enough that mgf_val is within
    quad's absolute tolerance of 1).

    Derivation: mgf_val = t1 + t2 + t3, where (f = f_Bounded_eta_RA, F its CDF)

        t1 = int_0^D  exp(omega*y)                 f(y) dy
        t2 = int_D^r  exp(omega*((1-C)y + C*D))     f(y) dy
        t3 = (1-F(r)) * exp(omega*((1-C)*r + C*D))

    Writing exp(x) = 1 + expm1(x) inside each term splits it into a
    "baseline probability mass" piece plus an expm1 correction:

        t1 = F(D)          + int_0^D  expm1(omega*y)                 f(y) dy
        t2 = (F(r)-F(D))    + int_D^r  expm1(omega*((1-C)y + C*D))     f(y) dy
        t3 = (1-F(r))       + (1-F(r)) * expm1(omega*((1-C)*r + C*D))

    The three baseline pieces sum to F(D) + (F(r)-F(D)) + (1-F(r)) = 1
    EXACTLY (by definition of a CDF), cancelling the "-1" analytically
    instead of in floating point, so

        mgf_val - 1 = int_0^D  expm1(omega*y)                 f(y) dy
                    + int_D^r  expm1(omega*((1-C)y + C*D))     f(y) dy
                    + (1-F(r)) * expm1(omega*((1-C)*r + C*D)).

    Only f_Bounded_eta_RA and a plain (omega-independent) survival
    probability for (1-F(r)) are needed; term_1/2/3's internals are
    irrelevant here. For C == 0 this collapses to a single integral over
    (0, r) plus the point-mass term at r, matching MGF_Bounded_Losses_RA's
    C == 0 branch.
    """
    f = lambda y: f_Bounded_eta_RA(dist_type, y, k=k, eta_0=eta_0, mu=mu, sigma=sigma)

    # (1 - F(r)): plain probability, no omega dependence -> no cancellation
    # risk regardless of how small omega is. Computed via direct
    # integration of the tail so we don't depend on a closed-form survival
    # function being available for both dist_types.
    one_minus_F_r, _ = quad(f, r, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)

    if C == 0:
        integrand = lambda y: np.expm1(omega * y) * f(y)
        term_int, _ = quad(integrand, 0, r, limit=200, epsabs=1e-12, epsrel=1e-12)
        term_r = one_minus_F_r * np.expm1(omega * r)
        return term_int + term_r

    integrand_1 = lambda y: np.expm1(omega * y) * f(y)
    term_1_corr, _ = quad(integrand_1, 0, D, limit=200, epsabs=1e-12, epsrel=1e-12)

    integrand_2 = lambda y: np.expm1(omega * ((1.0 - C) * y + C * D)) * f(y)
    term_2_corr, _ = quad(integrand_2, D, r, limit=200, epsabs=1e-12, epsrel=1e-12)

    term_3_corr = one_minus_F_r * np.expm1(omega * ((1.0 - C) * r + C * D))

    return term_1_corr + term_2_corr + term_3_corr


# =============================================================================
# PDE right-hand side (with the M/W_min clip on delta_W)
# =============================================================================

def compute_spatial_derivatives_Hawkes_RA(W, d_lambda, d_h):
    """Same N==1 safeguard as compute_lambda_derivative_Q: if the lambda
    grid has a single point (e.g. lambda_min == lambda_max), dW/dlambda is
    identically 0 there instead of indexing out of bounds."""
    N = W.shape[0]
    dW_dlambda = np.zeros_like(W)
    dW_dh = np.zeros_like(W)
    if N > 1:
        dW_dlambda[1:-1, :] = (W[2:, :] - W[:-2, :]) / (2.0 * d_lambda)
        dW_dlambda[0, :] = (W[1, :] - W[0, :]) / d_lambda
        dW_dlambda[-1, :] = (W[-1, :] - W[-2, :]) / d_lambda
    dW_dh[:, 1:-1] = (W[:, 2:] - W[:, :-2]) / (2.0 * d_h)
    dW_dh[:, 0] = (W[:, 1] - W[:, 0]) / d_h
    dW_dh[:, -1] = (W[:, -1] - W[:, -2]) / d_h
    return dW_dlambda, dW_dh


def PDE_rhs_Hawkes_RA(
    t, W_flat, N, M, lambda_grid, h_grid, d_lambda, d_h, n_tilde,
    rho, gamma, delta, v, alpha, xi, omega, mgf_val, S_grid,
    T_final, M_interp, logQ_interp,
    mgf_minus_1=None,
):
    W = W_flat.reshape((N, M))
    dW_dlambda, dW_dh = compute_spatial_derivatives_Hawkes_RA(W, d_lambda, d_h)

    n_indices = np.arange(N)
    n_shift = np.minimum(n_indices + n_tilde, N - 1)
    W_shift = W[n_shift, :]

    control_term_z = (np.maximum(dW_dh - delta, 0.0) ** 2) / (2.0 * gamma)

    delta_W = W_shift - W

    # Clip delta_W using the M(t,h) and W_min(t,lambda,h;C) bounds.
    # Numerical safeguard: keeps exp(-omega*delta_W) from blowing up
    # during transient solver steps. bound is (N, M): M_RA doesn't depend
    # on lambda, but W_min does (through Q), so the clipping window is no
    # longer flat across lambda the way it used to be.
    M_vals = M_interp(t)  # (M,)
    W_min_vals = W_min_closed_form(
        t, lambda_grid, h_grid, T_final, rho, gamma, delta, omega, logQ_interp
    )  # (N, M)
    bound = np.maximum(M_vals[None, :] - W_min_vals, 0.0)
    delta_W = np.clip(delta_W, -bound, bound)

    # Jump/premium term, written to avoid catastrophic cancellation as
    # omega -> 0. The naive form (lambda/omega)*(1 - ell*exp(-omega*dW))
    # subtracts two O(1) quantities that both -> 0 with omega, losing all
    # precision before the 1/omega amplification. Instead:
    #   1 - ell*exp(-omega*dW) = (1-ell) + ell*(-expm1(-omega*dW))
    # with (1-ell) = -S*mgf_minus_1 computed by _mgf_minus_1_RA without
    # ever subtracting two near-1 quantities either.
    S_mat = S_grid[None, :]
    if mgf_minus_1 is None:
        mgf_minus_1 = mgf_val - 1.0

    expm1_neg_omega_dW = np.expm1(-omega * delta_W)
    ell = (1.0 - S_mat) + S_mat * mgf_val
    one_minus_ell = -S_mat * mgf_minus_1

    bracket_term = one_minus_ell - ell * expm1_neg_omega_dW
    premium_term = (lambda_grid[:, None] / omega) * bracket_term

    dW_dt = (
        -xi * (alpha - lambda_grid[:, None]) * dW_dlambda
        + rho * h_grid[None, :] * dW_dh
        - control_term_z
        - premium_term
    )
    return dW_dt.flatten()


def Hawkes_PIDE_solver_RA(
    dist_type='lomax', k=3.0, eta_0=10.0, mu=None, sigma=None, C=0.8,
    omega=0.1, r=200.0, delta=1.0,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=0.1,
    M_K_steps=200, M_d_h=None,
    Q_K_steps=None,
    log_every_seconds=None,
):
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    N = len(lambda_grid)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    M = len(h_grid)

    W_T = np.tile(np.sqrt(h_grid), (N, 1))
    t_span = (T_final, t_0)
    t_eval = np.linspace(T_final, t_0, K_steps)

    S_grid = np.array([GLI(h, v, a, b) for h in h_grid])

    if d is not None:
        exp_eta_i_mean = eta_0 / (k - 1) if dist_type.lower() == 'lomax' else np.exp(mu + (sigma ** 2) / 2.0)
        D = d * exp_eta_i_mean

    mgf_val, mgf_minus_1 = MGF_Bounded_Losses_RA(
        dist_type=dist_type, C=C, omega=omega, r=r,
        k=k, eta_0=eta_0, mu=mu, sigma=sigma, D=D,
        return_minus_1=True,
    )

    n_tilde = int(np.floor(beta / d_lambda))

    M_d_h_use = d_h if M_d_h is None else M_d_h
    M_tensor, (h_grid_M, t_eval_M) = solve_M_upper_bound(
        t_0=t_0, T_final=T_final, K_steps=M_K_steps,
        h_min=h_min, h_max=h_max, d_h=M_d_h_use,
        rho=rho, gamma=gamma, delta=delta,
    )
    if M_d_h_use != d_h:
        rgi = RegularGridInterpolator((h_grid_M, t_eval_M[::-1]), M_tensor[:, ::-1],
                                       bounds_error=False, fill_value=None)

        def M_interp(t):
            pts = np.stack([h_grid, np.full_like(h_grid, t)], axis=1)
            return rgi(pts)
    else:
        M_interp = build_M_interpolator(M_tensor, h_grid_M, t_eval_M)

    # Q(t,lambda,h;C): drives the lower bound W_min through the log(Q)/omega
    # term (see W_min_closed_form). ell_grid reuses mgf_val computed above,
    # so Q is guaranteed to use the same C, omega, r, and distribution
    # parameters as the main W solve.
    ell_grid = (1.0 - S_grid) + S_grid * mgf_val
    Q_K_steps_use = K_steps if Q_K_steps is None else Q_K_steps
    logQ_tensor, (lambda_grid_Q, h_grid_Q, t_eval_Q) = solve_logQ_lower_bound(
        t_0=t_0, T_final=T_final, K_steps=Q_K_steps_use,
        lambda_min=lambda_min, lambda_max=lambda_max, d_lambda=d_lambda,
        h_min=h_min, h_max=h_max, d_h=d_h,
        xi=xi, alpha=alpha, n_tilde=n_tilde,
        ell_grid=ell_grid,
    )
    logQ_interp = build_logQ_interpolator(logQ_tensor, lambda_grid_Q, h_grid_Q, t_eval_Q)

    extra_args = (
        N, M, lambda_grid, h_grid, d_lambda, d_h, n_tilde,
        rho, gamma, delta, v, alpha, xi, omega, mgf_val, S_grid,
        T_final, M_interp, logQ_interp, mgf_minus_1,
    )

    rhs_to_use = PDE_rhs_Hawkes_RA
    if log_every_seconds is not None:
        rhs_to_use = make_logging_rhs(
            PDE_rhs_Hawkes_RA, t_0=t_0, T_final=T_final,
            log_every_seconds=log_every_seconds, label="RA",
        )

    jac_sparsity = build_jacobian_sparsity(N, M, n_tilde)

    sol = solve_ivp(
        fun=rhs_to_use, t_span=t_span, y0=W_T.flatten(), method='Radau',
        t_eval=t_eval, args=extra_args, rtol=1e-9, atol=1e-11, #rtol=1e-6, atol=1e-8
        jac_sparsity=jac_sparsity,
    )

    if not sol.success:
        raise RuntimeError(
            f"solve_ivp did not converge in Hawkes_PIDE_solver_RA (status={sol.status}): "
            f"{sol.message}\nCompleted {sol.y.shape[1]} of {len(t_eval)} time points."
        )

    W_tensor = sol.y.reshape((N, M, len(t_eval)))
    return W_tensor, (N, M, t_eval, lambda_grid, h_grid)


def solve_W_by_C_RA(
    C_grid, dist_type, k=None, eta_0=None, mu=None, sigma=None,
    omega=0.1, r=200.0, delta=1.0,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=None,
    M_K_steps=200, M_d_h=None, log_every_seconds=None,
):
    """
    Part 1 of the optimal-C decision map (split from the former
    optimal_C_decision_map_RA so the expensive PIDE solve -- one
    Hawkes_PIDE_solver_RA call per C, the only step that depends on
    omega/r/dist params -- can be reused across many premium
    configurations (theta, Principle) without resolving W. Pass this
    function's return value straight into plot_optimal_C_decision_map_RA;
    sweeping theta there is then free.

    Returns a dict with:
      - 'W_by_C':  {C: W(t=0,lambda,h;C)} for each C in C_grid
      - 'lambda_grid', 'h_grid'
      - 'context': static params plot_optimal_C_decision_map_RA needs for
                    the premium calc (Initial_Net_Value_Hawkes_RA) and for
                    the plot title, bundled here so callers don't have to
                    remember and re-pass every solver parameter twice.
    """
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)

    W_by_C = {}
    for C in C_grid:
        print(f"Solving W for C={C} ...")
        W_tensor, _ = Hawkes_PIDE_solver_RA(
            dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=C,
            omega=omega, r=r, delta=delta,
            t_0=t_0, T_final=T_final, K_steps=K_steps,
            h_min=h_min, h_max=h_max, d_h=d_h,
            lambda_min=lambda_min, lambda_max=lambda_max, d_lambda=d_lambda,
            rho=rho, gamma=gamma, v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta,
            D=D, d=d, M_K_steps=M_K_steps, M_d_h=M_d_h, log_every_seconds=log_every_seconds,
        )
        W_by_C[C] = W_tensor[:, :, -1].copy()  # W(t=0, lambda, h)

    context = dict(
        dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma,
        v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta, D=D, d=d,
        T_final=T_final, r=r,
    )

    return {
        'W_by_C': W_by_C,
        'lambda_grid': lambda_grid,
        'h_grid': h_grid,
        'context': context,
    }

# =============================================================================
# Evaluation of W under a static investment strategy
# ============================================================================

def PDE_rhs_W_zbar(
    t, W_flat, N, M, lambda_grid, h_grid, d_lambda, d_h, n_tilde,
    rho, gamma, delta, alpha, xi, omega, mgf_val, S_grid,
    z_bar, mgf_minus_1=None,
):
    """
    Same reduced PIDE as PDE_rhs_Hawkes_RA, but with a FIXED constant
    control z_bar over [t,T] instead of the sup_z{...} optimization. This
    is the risk-averse analog of _static_control_objective: it evaluates
    W_zbar(t,lambda,h;C,z_bar), the certainty equivalent under committing
    to z_bar for the rest of the horizon, rather than the dynamically
    optimal W.

    Two things drop out relative to PDE_rhs_Hawkes_RA, not just the
    control term:
      1. control_term_z is the closed-form value of z*d(W)/dh - delta*z -
         gamma*z^2/2 evaluated AT z_bar (no max/threshold -- z_bar can be
         negative in principle, though economically you'd only ever pass
         z_bar >= 0).
      2. The M(t,h)/W_min(t,lambda,h;C) clipping on delta_W is NOT reused
         here. That bound was derived specifically for the z=rho*h
         bounding policy (see Section 4.3.1) -- it is NOT a valid bound
         for an arbitrary fixed z_bar, and reusing it would silently bias
         the result. The transient instability that motivated clipping in
         PDE_rhs_Hawkes_RA came from the control argmax switching back and
         forth during Newton iterations; with z_bar fixed, that source of
         instability isn't present, so no clip is applied here. If you do
         see real numerical trouble for some z_bar, that needs a bound
         derived for THAT specific z_bar, not a reuse of the rho*h one.
    """
    W = W_flat.reshape((N, M))
    dW_dlambda, dW_dh = compute_spatial_derivatives_Hawkes_RA(W, d_lambda, d_h)

    n_indices = np.arange(N)
    n_shift = np.minimum(n_indices + n_tilde, N - 1)
    W_shift = W[n_shift, :]

    # --- the only real change vs. the optimized-control term ---
    control_term_z = z_bar * dW_dh - delta * z_bar - 0.5 * gamma * (z_bar ** 2)

    delta_W = W_shift - W  # no clipping here, see docstring

    S_mat = S_grid[None, :]
    if mgf_minus_1 is None:
        mgf_minus_1 = mgf_val - 1.0

    expm1_neg_omega_dW = np.expm1(-omega * delta_W)
    ell = (1.0 - S_mat) + S_mat * mgf_val
    one_minus_ell = -S_mat * mgf_minus_1

    bracket_term = one_minus_ell - ell * expm1_neg_omega_dW
    premium_term = (lambda_grid[:, None] / omega) * bracket_term

    dW_dt = (
        -xi * (alpha - lambda_grid[:, None]) * dW_dlambda
        + rho * h_grid[None, :] * dW_dh
        - control_term_z
        - premium_term
    )
    return dW_dt.flatten()


def solve_W_zbar_RA(
    z_bar, dist_type='lomax', k=3.0, eta_0=10.0, mu=None, sigma=None, C=0.8,
    omega=0.1, r=200.0, delta=1.0,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=0.1,
    log_every_seconds=None,
):
    """
    Solves W_zbar(t,lambda,h;C,z_bar) -- the risk-averse analog of
    compute_static_trajectory_for_pair / evaluate_J_Hawkes_constant_policy,
    but for CARA: unlike the risk-neutral case (a cheap closed-form
    integral per (t,lambda,h) triple), this needs the full PIDE solved on
    the whole (lambda,h) grid, once per z_bar -- see the discussion on why
    that's structurally unavoidable under CARA (the jump term's
    exponential nonlinearity doesn't go away just because z is fixed).

    Notably CHEAPER than Hawkes_PIDE_solver_RA: no M(t,h) solve, no
    Q(t,lambda,h;C) solve -- neither is needed since there's no clipping
    here (see PDE_rhs_W_zbar's docstring).
    """
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    N = len(lambda_grid)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    M = len(h_grid)

    W_T = np.tile(np.sqrt(h_grid), (N, 1))
    t_span = (T_final, t_0)
    t_eval = np.linspace(T_final, t_0, K_steps)

    S_grid = np.array([GLI(h, v, a, b) for h in h_grid])

    if d is not None:
        exp_eta_i_mean = eta_0 / (k - 1) if dist_type.lower() == 'lomax' else np.exp(mu + (sigma ** 2) / 2.0)
        D = d * exp_eta_i_mean

    mgf_val, mgf_minus_1 = MGF_Bounded_Losses_RA(
        dist_type=dist_type, C=C, omega=omega, r=r,
        k=k, eta_0=eta_0, mu=mu, sigma=sigma, D=D,
        return_minus_1=True,
    )

    n_tilde = int(np.floor(beta / d_lambda))

    extra_args = (
        N, M, lambda_grid, h_grid, d_lambda, d_h, n_tilde,
        rho, gamma, delta, alpha, xi, omega, mgf_val, S_grid,
        z_bar, mgf_minus_1,
    )

    rhs_to_use = PDE_rhs_W_zbar
    if log_every_seconds is not None:
        rhs_to_use = make_logging_rhs(
            PDE_rhs_W_zbar, t_0=t_0, T_final=T_final,
            log_every_seconds=log_every_seconds, label=f"RA z_bar={z_bar:g}",
        )

    jac_sparsity = build_jacobian_sparsity(N, M, n_tilde)

    sol = solve_ivp(
        fun=rhs_to_use, t_span=t_span, y0=W_T.flatten(), method='Radau',
        t_eval=t_eval, args=extra_args, rtol=1e-6, atol=1e-8,
        jac_sparsity=jac_sparsity,
    )

    if not sol.success:
        raise RuntimeError(
            f"solve_ivp did not converge in solve_W_zbar_RA (z_bar={z_bar}, status={sol.status}): "
            f"{sol.message}\nCompleted {sol.y.shape[1]} of {len(t_eval)} time points."
        )

    W_tensor = sol.y.reshape((N, M, len(t_eval)))
    return W_tensor, (N, M, t_eval, lambda_grid, h_grid)

# =============================================================================
# Net value / certainty-equivalent wealth
# =============================================================================

def Initial_Net_Value_Hawkes_RA(
    W, lambda_grid, h_grid, T_final, dist_type,
    k=None, eta_0=None, mu=None, sigma=None, C=None,
    theta=0.3, v=0.65, a=0.1, b=1.0, alpha=27.0, xi=15.0, beta=9.0,
    D=None, d=None, Principle=None
):
    # W arrives already 2D (N,M) -- no time dimension.
    init_W = W.copy()
    dist_type = dist_type.lower()
    if dist_type == 'lomax':
        if k is None or eta_0 is None:
            raise ValueError("Parameters 'k' and 'eta_0' are required for Lomax.")
        calc_premium = lambda l_val, h_val: calculate_premium_lomax(
            t=T_final, k=k, eta_0=eta_0, C=C, theta=theta, v=v, a=a, b=b,
            alpha=alpha, xi=xi, beta=beta, H_0=h_val, lambda_0=l_val,
            D=D, d=d, Principle=Principle, process_type='hawkes'
        )
    elif dist_type == 'lognormal':
        if mu is None or sigma is None:
            raise ValueError("Parameters 'mu' and 'sigma' are required for Lognormal.")
        calc_premium = lambda l_val, h_val: calculate_premium_lognormal(
            t=T_final, mu=mu, sigma=sigma, C=C, theta=theta, v=v, a=a, b=b,
            alpha=alpha, xi=xi, beta=beta, H_0=h_val, lambda_0=l_val,
            D=D, d=d, Principle=Principle, process_type='hawkes'
        )
    else:
        raise ValueError(f"Unsupported distribution '{dist_type}'. Choose 'lomax' or 'lognormal'.")
    for i, lambda_val in enumerate(lambda_grid):
        for j, h_val in enumerate(h_grid):
            init_W[i, j] -= calc_premium(lambda_val, h_val)
    return init_W


def V_RA(W_tensor, x, omega, dist_type, lambda_min=27, lambda_max=216, d_lambda=1,
    h_min=0, h_max=50, d_h=0.5, T_final=1,
    k=None, eta_0=None, mu=None, sigma=None, C=None,
    theta=0.3, v=0.65, a=0.1, b=1.0, alpha=27.0, xi=15.0, beta=9.0,
    D=None, d=None, Principle=None
):
    """Returns the certainty-equivalent argument x + W(0,lambda,h) - Pi(lambda,h)
    -- NOT the exponentiated V = -e^{-omega*(...)}. Since -e^{-omega*u} is
    monotonic increasing in u, comparing this linear quantity across C (or
    across policies in general) gives the same argmax as comparing V itself,
    and x cancels out of any such comparison. Kept unexponentiated here
    because that is what every downstream use (C* selection, insurance
    decision maps) actually needs."""
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    N = len(lambda_grid)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    M = len(h_grid)

    W_t_0 = W_tensor[:, :, -1].copy()

    # Pass a zero template: Initial_Net_Value_Hawkes_RA subtracts the premium
    # from whatever it receives, so zeros in -> exactly -Pi(lambda,h) out.
    Pi_template = np.zeros((N, M))
    neg_Pi_matrix = Initial_Net_Value_Hawkes_RA(
        W=Pi_template, lambda_grid=lambda_grid, h_grid=h_grid, T_final=T_final, dist_type=dist_type,
        k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=C,
        theta=theta, v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta,
        D=D, d=d, Principle=Principle
    )

    initial_wealth_x = x + neg_Pi_matrix  # x - Pi(lambda,h)
    V = initial_wealth_x + W_t_0
    return V
