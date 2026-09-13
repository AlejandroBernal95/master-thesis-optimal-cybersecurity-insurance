"""
Risk-neutral HJB-PIDE for the joint cybersecurity-investment / cyberinsurance
problem, under both Hawkes and Poisson attack-arrival dynamics.

State: (t, lambda, h) for Hawkes; (t, h) for Poisson (constant intensity).
Terminal condition: V(T, ., h) = sqrt(h).
Control: security investment rate z, optimal z* = max(dV/dh - 1, 0) / gamma.

Visualization utilities (plot_value_function, plot_optimal_control,
initial_comparison_1_0_Hawkes) live in plotting.py. Independent verification
of the optimal control (static-control benchmark, Poisson closed-loop
evaluation) lives in validation.py.
"""

import numpy as np
from scipy.integrate import solve_ivp

from premiums import GLI, calculate_premium_lomax, calculate_premium_lognormal
from pide_common import build_jacobian_sparsity


# =============================================================================
# Loss terms (shared by the Hawkes and Poisson RHS)
# =============================================================================

def compute_loss_terms(dist_type, C, D, S_grid, k=None, eta_0=None, mu=None, sigma=None):

    dist_type = dist_type.lower()

    if dist_type == 'lomax':
        if k is None or eta_0 is None:
            raise ValueError("Parameters 'k' and 'eta_0' are required for Lomax distribution.")
        if k <= 2:
            raise ValueError("The k parameter must be larger than 2.")

        # E[eta_i] = eta_0 / (k - 1) under Lomax/Pareto II
        exp_eta_i = eta_0 / (k - 1)

        # With C=0 the retention term is multiplied by C and vanishes identically
        # regardless of D -- skip computing it (D may even be an unrelated/dummy
        # value in that case, since it plays no role in the result).
        if C == 0:
            return exp_eta_i, exp_eta_i * S_grid

        retention_factor = (eta_0 / (eta_0 + D)) ** (k - 1)
        term_S_vec = exp_eta_i * (1.0 - C * retention_factor) * S_grid

    elif dist_type == 'lognormal':
        if mu is None or sigma is None:
            raise ValueError("Parameters 'mu' and 'sigma' are required for Lognormal distribution.")

        # E[eta_i] = exp(mu + sigma^2/2) under Lognormal
        exp_eta_i = np.exp(mu + (sigma ** 2) / 2.0)

        if C == 0:
            return exp_eta_i, exp_eta_i * S_grid

        d_LN = (mu - np.log(D)) / sigma
        term_S_vec = (exp_eta_i + C * (D * norm_cdf(d_LN) - exp_eta_i * norm_cdf(d_LN + sigma))) * S_grid

    else:
        raise ValueError(f"Unsupported distribution type '{dist_type}'. Choose 'lomax' or 'lognormal'.")

    return exp_eta_i, term_S_vec


from scipy.stats import norm as _norm
norm_cdf = _norm.cdf


def _resolve_deductible(dist_type, C, D, d, k=None, eta_0=None, mu=None, sigma=None):
    """Shared D/d resolution used by both PIDE solvers. Kept required (even
    when C=0) because calculate_premium_* downstream (in Initial_Net_Value_*)
    still needs a valid D/d -- see module docstring notes on ghost inputs."""
    if d is not None:
        if not (0 <= d <= 1):
            raise ValueError("d has to be in the interval [0, 1].")
        exp_eta_i_mean = eta_0 / (k - 1) if dist_type.lower() == 'lomax' else np.exp(mu + (sigma ** 2) / 2.0)
        return d * exp_eta_i_mean
    elif D < 0:
        raise ValueError("D (Deductible) must be non-negative.")
    return D


# =============================================================================
# Hawkes Process Loss Modeling
# =============================================================================

def Initial_Net_Value_Hawkes(
    V, lambda_grid, h_grid, T_final, dist_type,
    k=None, eta_0=None, mu=None, sigma=None, C=None,
    theta=0.3, v=0.65, a=0.1, b=1.0, alpha=27.0, xi=15.0, beta=9.0,
    D=None, d=None, Principle=None
):
    init_v = V[:, :, -1].copy()
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
            init_v[i, j] -= calc_premium(lambda_val, h_val)

    return init_v


def compute_spatial_derivatives_Hawkes(V, d_lambda, d_h):

    dV_dlambda = np.zeros_like(V)
    dV_dh = np.zeros_like(V)

    dV_dlambda[1:-1, ...] = (V[2:, ...] - V[:-2, ...]) / (2.0 * d_lambda)
    dV_dh[:, 1:-1, ...] = (V[:, 2:, ...] - V[:, :-2, ...]) / (2.0 * d_h)

    dV_dlambda[0, ...] = (V[1, ...] - V[0, ...]) / d_lambda
    dV_dlambda[-1, ...] = (V[-1, ...] - V[-2, ...]) / d_lambda
    dV_dh[:, 0, ...] = (V[:, 1, ...] - V[:, 0, ...]) / d_h
    dV_dh[:, -1, ...] = (V[:, -1, ...] - V[:, -2, ...]) / d_h

    return dV_dlambda, dV_dh


def PDE_rhs_Hawkes(
    t, V_flat,
    N, M, lambda_grid, h_grid, d_lambda, d_h, n_tilde,
    rho, gamma, v, alpha, xi, exp_eta_i, term_S_vec
):

    V = V_flat.reshape((N, M))
    dV_dlambda, dV_dh = compute_spatial_derivatives_Hawkes(V, d_lambda, d_h)

    n_indices = np.arange(N)
    n_shift = np.minimum(n_indices + n_tilde, N - 1)
    V_shift = V[n_shift, :]

    control_term_z = (np.maximum(dV_dh - 1.0, 0.0) ** 2) / (2.0 * gamma)
    net_loss_lambda = lambda_grid[:, None] * (exp_eta_i * v - term_S_vec)

    dV_dt = (
        xi * (lambda_grid[:, None] - alpha) * dV_dlambda
        + rho * h_grid * dV_dh
        - lambda_grid[:, None] * (V_shift - V)
        - net_loss_lambda
        - control_term_z
    )

    return dV_dt.flatten()


def Hawkes_PIDE_solver(
    dist_type=None, k=None, eta_0=None, mu=None, sigma=None, C=None,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=None, U_sqrt = True,
):

    if not (0 <= C <= 1):
        raise ValueError("C has to be in the interval [0, 1].")
    
    if C == 0:
        D = 0
    else:
        if (D is None) == (d is None):
            raise ValueError("Exactly one of 'D' or 'd' must be specified (the other must be None).")

    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    N = len(lambda_grid)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    M = len(h_grid)

    V_T = np.zeros((N, M))
    if U_sqrt == True:
        for m in range(M):
            V_T[:, m] = np.sqrt(h_grid[m])
    V_init_flat = V_T.flatten()

    t_span = (T_final, t_0)
    t_eval = np.linspace(T_final, t_0, K_steps)

    S_grid = np.array([GLI(h, v, a, b) for h in h_grid])
    D = _resolve_deductible(dist_type, C, D, d, k=k, eta_0=eta_0, mu=mu, sigma=sigma)

    exp_eta_i, term_S_vec = compute_loss_terms(
        dist_type=dist_type, C=C, D=D, S_grid=S_grid, k=k, eta_0=eta_0, mu=mu, sigma=sigma
    )

    n_tilde = int(np.floor(beta / d_lambda))

    extra_args = (
        N, M, lambda_grid, h_grid, d_lambda, d_h, n_tilde,
        rho, gamma, v, alpha, xi, exp_eta_i, term_S_vec
    )

    # Same grid topology (5-point stencil in lambda/h + the n_tilde jump shift)
    # as the risk-averse solver, so the same sparsity builder applies unchanged.
    jac_sparsity = build_jacobian_sparsity(N, M, n_tilde)

    sol = solve_ivp(
        fun=PDE_rhs_Hawkes, t_span=t_span, y0=V_init_flat, method='Radau',
        t_eval=t_eval, args=extra_args, vectorized=False, jac_sparsity=jac_sparsity, rtol=1e-8, atol=1e-10,
    )

    V_tensor = sol.y.reshape((N, M, len(t_eval)))
    return V_tensor, (N, M, t_eval, lambda_grid, h_grid)


def z_star(V, d_lambda, d_h, gamma):
    dV_dlambda, dV_dh = compute_spatial_derivatives_Hawkes(V, d_lambda, d_h)
    return np.maximum(dV_dh - 1.0, 0.0) / gamma


# =============================================================================
# Poisson Process Loss Modeling
# =============================================================================
# NOTE: Initial_Net_Value_Poisson moved to discarded_experiments.py -- its
# only caller (initial_comparison_1_0_Poisson) lives there too.

def compute_spatial_derivatives_Poisson(V, d_h):
    dV_dh = np.zeros_like(V)
    dV_dh[1:-1, ...] = (V[2:, ...] - V[:-2, ...]) / (2.0 * d_h)
    dV_dh[0, ...] = (V[1, ...] - V[0, ...]) / d_h
    dV_dh[-1, ...] = (V[-1, ...] - V[-2, ...]) / d_h
    return dV_dh


def PDE_rhs_Poisson(t, V, M, h_grid, d_h, rho, gamma, v, lambda_0, exp_eta_i, term_S_vec):

    dV_dh = compute_spatial_derivatives_Poisson(V, d_h)
    control_term_z = (np.maximum(dV_dh - 1.0, 0.0) ** 2) / (2.0 * gamma)
    net_loss = lambda_0 * (exp_eta_i * v - term_S_vec)

    return rho * h_grid * dV_dh - net_loss - control_term_z


def Poisson_PDE_solver(
    dist_type=None, k=None, eta_0=None, mu=None, sigma=None, C=None,
    lambda_0=27.0, t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0, D=None, d=None,
):

    if not (0 <= C <= 1):
        raise ValueError("C has to be in the interval [0, 1].")
    
    if C == 0:
        D = 0
    else:
        if (D is None) == (d is None):
            raise ValueError("Exactly one of 'D' or 'd' must be specified (the other must be None).")

    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    M = len(h_grid)

    V_T = np.sqrt(h_grid)
    t_span = (T_final, t_0)
    t_eval = np.linspace(T_final, t_0, K_steps)

    S_grid = np.array([GLI(h, v, a, b) for h in h_grid])
    D = _resolve_deductible(dist_type, C, D, d, k=k, eta_0=eta_0, mu=mu, sigma=sigma)

    exp_eta_i, term_S_vec = compute_loss_terms(
        dist_type=dist_type, C=C, D=D, S_grid=S_grid, k=k, eta_0=eta_0, mu=mu, sigma=sigma
    )

    extra_args = (M, h_grid, d_h, rho, gamma, v, lambda_0, exp_eta_i, term_S_vec)

    # 1D in h only (no lambda dimension, no jump term) -- the dense finite-
    # difference Jacobian solve_ivp estimates by default is already cheap
    # here, so this solver does not need jac_sparsity.
    sol = solve_ivp(
        fun=PDE_rhs_Poisson, t_span=t_span, y0=V_T, method='Radau',
        t_eval=t_eval, args=extra_args, vectorized=False
    )

    return sol.y, (M, t_eval, h_grid)


def z_star_Poisson(V, d_h, gamma):
    dV_dh = compute_spatial_derivatives_Poisson(V, d_h)
    return np.maximum(dV_dh - 1.0, 0.0) / gamma