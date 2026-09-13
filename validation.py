"""
Independent verification of the risk-neutral PIDE solvers' optimal control,
by two methods that never touch the PIDE machinery itself:

1. Hawkes static-control benchmark: re-optimizes a CONSTANT z over each
   remaining horizon [t_k,T] via direct numerical optimization
   (differential_evolution), instead of solving the PIDE. Comparing this
   "best constant control" trajectory against the PIDE's dynamically-optimal
   V(t,lambda,h) both cross-validates the PIDE solution and quantifies the
   value of dynamic vs. static control (see plot_dynamic_vs_static_gain /
   plot_dynamic_vs_constant_gain in plotting.py).

2. Poisson closed-loop evaluation: since lambda is constant under Poisson,
   integrates the deterministic closed-loop ODE forward under the extracted
   feedback policy z*(t,h) from Poisson_PDE_solver -- an exact (not Monte
   Carlo) validation of the extracted control, up to ODE-integration
   tolerance.

Depends on pide_risk_neutral.py for the shared loss-rate machinery
(compute_loss_terms, _resolve_deductible) but is not needed to run the PIDE
solvers themselves -- this module is validation tooling, not the pipeline.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import differential_evolution
from scipy.interpolate import RegularGridInterpolator

from processes import Expectation_Lambda_t
from premiums import GLI
from pide_risk_neutral import compute_loss_terms, _resolve_deductible


# =============================================================================
# Static-control benchmark: independent verification of the PIDE's optimal
# control, by re-optimizing a CONSTANT z over each remaining horizon [t_k,T]
# via direct numerical optimization instead of solving the PIDE.
# =============================================================================

def _static_control_integrand(s, t_k, l_val, h_val, z_bar, rho, alpha, xi, beta,
                               dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b):
    """Integrand of the running loss-rate term in J(t_k,lambda,h;z_bar)
    (thesis eq. 20), evaluated along the deterministic h-trajectory induced
    by holding the control fixed at z_bar over [t_k, T]."""
    E_l = Expectation_Lambda_t(t=s - t_k, alpha=alpha, lambda_0=l_val, xi=xi, beta=beta)
    H_s = h_val * np.exp(-rho * (s - t_k)) + (z_bar / rho) * (1.0 - np.exp(-rho * (s - t_k)))
    S_val = GLI(H_s, v, a, b)
    exp_eta_i, term_S_vec = compute_loss_terms(
        dist_type=dist_type, C=C, D=D_val, S_grid=np.array([S_val]),
        k=k, eta_0=eta_0, mu=mu, sigma=sigma,
    )
    return E_l * (exp_eta_i * v - term_S_vec[0])


def _static_control_objective(z_array, t_k, T_final, l_val, h_val, rho, gamma,
                               alpha, xi, beta, dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b):
    """Negative of J(t_k,lambda,h;z_bar) (thesis eq. 20) for a constant
    control z_bar held over [t_k,T] -- negated because differential_evolution
    minimizes."""
    z_bar = z_array[0]
    tau = T_final - t_k
    integrand_args = (t_k, l_val, h_val, z_bar, rho, alpha, xi, beta,
                       dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b)
    int_val, _ = quad(_static_control_integrand, t_k, T_final, args=integrand_args,
                   epsabs=1e-10, epsrel=1e-10, limit=200)
    cost = tau * (z_bar + (gamma / 2.0) * (z_bar ** 2))
    H_T = h_val * np.exp(-rho * tau) + (z_bar / rho) * (1.0 - np.exp(-rho * tau))
    return -(int_val - cost + np.sqrt(H_T))


def compute_static_trajectory_for_pair(
    l_val, h_val, dist_type="lomax", k=2.1, eta_0=11, mu=None, sigma=None, C=0,
    t_eval=None, T_final=1.0, rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=0.1, z_bounds=(0.0, 50.0),
):
    """
    Computes J(t,lambda,h;z_bar*(t)) (thesis eq. 20-21) for each t in t_eval,
    where z_bar*(t_k) is found by re-optimizing a CONSTANT control over
    [t_k,T] via direct numerical optimization (differential_evolution) at
    each t_k independently -- a method fully independent of the PIDE solve.
    Comparing this "best constant control" trajectory against the PIDE's
    dynamically-optimal V(t,lambda,h) (which re-optimizes z continuously)
    both cross-validates the PIDE solution and quantifies the value of
    dynamic vs. static control -- see plot_dynamic_vs_static_gain in
    plotting.py.
    """
    if d is not None:
        exp_eta_i_mean = eta_0 / (k - 1) if dist_type.lower() == "lomax" else np.exp(mu + (sigma ** 2) / 2.0)
        D_val = d * exp_eta_i_mean
    else:
        D_val = D

    J_values = np.zeros(len(t_eval))
    for idx_t, t_k in enumerate(t_eval):
        tau = T_final - t_k
        if tau <= 1e-12:
            J_values[idx_t] = np.sqrt(h_val)
            continue
        objective_args = (t_k, T_final, l_val, h_val, rho, gamma, alpha, xi, beta,
                           dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b)
        res = differential_evolution(_static_control_objective, bounds=[z_bounds],
                                      args=objective_args, popsize=30, tol=1e-8)
        J_values[idx_t] = -res.fun

    return J_values


def evaluate_J_Hawkes_constant_policy(
    z_val, l_val, h_val, dist_type, k=None, eta_0=None, mu=None, sigma=None, C=0,
    t_eval=None, T_final=1.0, rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=0.1,
):
    """
    Computes J(t,lambda,h;z_val) (thesis eq. 20) for each t in t_eval, under
    a FIXED CONSTANT control z_val held over [t,T] -- the Hawkes-model analog
    of evaluate_J_Poisson_optimal_policy, but for an arbitrary given constant
    instead of the dynamic optimal feedback policy. Unlike
    compute_static_trajectory_for_pair (which searches for the BEST constant
    via differential_evolution), this evaluates the SAME underlying integral
    (_static_control_objective, already validated) at the z_val you pass, with
    no optimization -- so you can see how J behaves at, e.g., z=0 ("never
    invest") rather than only at the optimum.
    """
    if d is not None:
        exp_eta_i_mean = eta_0 / (k - 1) if dist_type.lower() == "lomax" else np.exp(mu + (sigma ** 2) / 2.0)
        D_val = d * exp_eta_i_mean
    else:
        D_val = D

    J_values = np.zeros(len(t_eval))
    for idx_t, t_k in enumerate(t_eval):
        tau = T_final - t_k
        if tau <= 1e-12:
            J_values[idx_t] = np.sqrt(h_val)
            continue
        objective_args = (t_k, T_final, l_val, h_val, rho, gamma, alpha, xi, beta,
                           dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b)
        J_values[idx_t] = -_static_control_objective(np.array([z_val]), *objective_args)

    return J_values


# =============================================================================
# Poisson closed-loop evaluation: independent verification of the PIDE's
# optimal control, by integrating the deterministic closed-loop ODE under the
# feedback policy z*(t,h) forward from t=0 to T.
# =============================================================================

def _poisson_closed_loop_rhs(t, state, z_interp, dist_type, C, D_val, k, eta_0, mu, sigma,
                              v, a, b, rho, gamma, lambda_0):
    """RHS of the augmented closed-loop ODE state=[H, J_accum] under the
    Poisson-optimal feedback policy z*(t,H) -- see
    evaluate_J_Poisson_optimal_policy for the sign derivation."""
    H, _ = state
    z_val = z_interp([[H, t]])[0]
    S_val = GLI(H, v, a, b)
    exp_eta_i, term_S_vec = compute_loss_terms(
        dist_type=dist_type, C=C, D=D_val, S_grid=np.array([S_val]),
        k=k, eta_0=eta_0, mu=mu, sigma=sigma,
    )
    net_loss = lambda_0 * (exp_eta_i * v - term_S_vec[0])
    dH_dt = z_val - rho * H
    dJ_dt = -(z_val + (gamma / 2.0) * z_val ** 2) + net_loss
    return [dH_dt, dJ_dt]


def evaluate_J_Poisson_optimal_policy(
    z_star_matrix, h_0, dist_type, k=None, eta_0=None, mu=None, sigma=None, C=None,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    lambda_0=27.0, D=None, d=None,
):
    """
    Evaluates J(0, h_0; z*) under the Poisson-optimal feedback policy z*(t,h)
    (thesis sec. 5.4): since lambda is constant under Poisson, H_t evolves as
    a deterministic ODE dH/dt = z*(t,H) - rho*H under ANY policy -- no
    stochastic averaging needed, unlike the Hawkes case. Integrates the
    closed-loop ODE forward from t=0 to T with an augmented state that
    accumulates the running reward, then adds sqrt(H_T). This should match
    V(0,h_0) from Poisson_PDE_solver up to ODE-integration tolerance -- an
    exact (not Monte Carlo) validation of the extracted control.

    Sign note: along the optimal trajectory, d/dt V(t,H*_t) =
    [z* + (gamma/2)(z*)^2] - net_loss(H*_t) (derived via chain rule from the
    validated PDE_rhs_Poisson -- the rho*h*dV/dh terms cancel exactly against
    the state's own drift). Integrating forward, net_loss enters the
    accumulated reward with a PLUS sign.
    """
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    t_eval = np.linspace(T_final, t_0, K_steps)

    t_incr = t_eval[::-1]
    z_incr = z_star_matrix[:, ::-1]
    z_interp = RegularGridInterpolator((h_grid, t_incr), z_incr, bounds_error=False, fill_value=None)

    D_val = _resolve_deductible(dist_type, C, D, d, k=k, eta_0=eta_0, mu=mu, sigma=sigma)

    rhs_args = (z_interp, dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b, rho, gamma, lambda_0)
    sol = solve_ivp(_poisson_closed_loop_rhs, (t_0, T_final), [h_0, 0.0], method="RK45",
                     args=rhs_args, rtol=1e-8, atol=1e-10, dense_output=True)

    H_T, J_accum_T = sol.y[:, -1]
    J_value = J_accum_T + np.sqrt(H_T)
    return J_value, sol


def evaluate_J_Poisson_optimal_policy_grid(
    z_star_matrix, dist_type, k=None, eta_0=None, mu=None, sigma=None, C=None,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    lambda_0=27.0, D=None, d=None,
):
    """Extends evaluate_J_Poisson_optimal_policy from a single (0,h_0) point
    to the full (h,t) grid: for each grid point (t_k, h_j), integrates the
    closed-loop ODE forward from (t_k, h_j) to T under the SAME fixed
    feedback policy z*(t,h). Returns J with the same shape/orientation as
    the PDE's own V (M, K_steps, decreasing t_eval)."""
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    t_eval = np.linspace(T_final, t_0, K_steps)

    t_incr = t_eval[::-1]
    z_incr = z_star_matrix[:, ::-1]
    z_interp = RegularGridInterpolator((h_grid, t_incr), z_incr, bounds_error=False, fill_value=None)

    D_val = _resolve_deductible(dist_type, C, D, d, k=k, eta_0=eta_0, mu=mu, sigma=sigma)
    rhs_args = (z_interp, dist_type, C, D_val, k, eta_0, mu, sigma, v, a, b, rho, gamma, lambda_0)

    J_matrix = np.empty((len(h_grid), K_steps))
    for idx_t, t_k in enumerate(t_eval):
        tau = T_final - t_k
        for idx_h, h_j in enumerate(h_grid):
            if tau <= 1e-9:
                J_matrix[idx_h, idx_t] = np.sqrt(h_j)
                continue
            sol = solve_ivp(_poisson_closed_loop_rhs, (t_k, T_final), [h_j, 0.0],
                             method="RK45", args=rhs_args, rtol=1e-7, atol=1e-9)
            H_T, J_accum_T = sol.y[:, -1]
            J_matrix[idx_h, idx_t] = J_accum_T + np.sqrt(H_T)

    return J_matrix, (t_eval, h_grid)
