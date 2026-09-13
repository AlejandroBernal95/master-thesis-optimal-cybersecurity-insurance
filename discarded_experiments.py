"""
Functional code that is NOT used by any cell in Thesis_Experiments.ipynb and
is NOT part of the thesis' final results. Kept here (rather than deleted)
purely for reference, in two groups:

1. Pareto Type I loss severity -- an earlier distribution choice, superseded
   by Lomax and Lognormal (the two distributions the thesis actually
   contributes). Premium/elasticity formulas + their 3D visualizations.

2. The Poisson-only insurance-decision framework (Initial_Net_Value_Poisson,
   initial_comparison_1_0_Poisson) and the older single-panel 3D comparison
   of Hawkes vs. Poisson value functions (compare_Hawkes_vs_Poisson_ht) --
   superseded in the notebook by plot_value_function_vs_poisson /
   _plot_hawkes_vs_poisson_grid (plotting.py), which is what the thesis
   figures actually use.

Everything here was verified to run (see the reachability check run when
this file was split out); "unused" means "not called by
Thesis_Experiments.ipynb", not "broken".
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from processes import Var_Nt, Expectation_Nt
from premiums import (
    GLI, _validate_C_and_deductible, _apply_premium_principle,
    calculate_premium_lomax, calculate_premium_lognormal,
)
from plotting import _save_figure, _save_panels, _format_ded_principle_str, _plot_3d_param_surface, _validate_tensor_arg
from pide_risk_neutral import Poisson_PDE_solver


# =============================================================================
# Process moments -- standalone Var(N_t) plot (superseded by Visual_Process,
# which already reports process trajectories; kept here since it's an
# independent, simpler view of the same quantity)
# =============================================================================

def Var_Nt_plot(process_type='hawkes', T_final=1, alpha=27, lambda_0=27, xi=15, beta=9, save_name=None):

    x = np.linspace(0, T_final, 500)
    y = Var_Nt(x, alpha, lambda_0, xi, beta, process_type=process_type)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Var$(N_t)$", color="blue", linewidth=1.5)
    plt.xlabel('$t$ (years)')
    plt.ylabel('$Var(N_t)$')
    plt.legend(loc='upper left')
    plt.grid(True)
    _save_figure(fig, save_name)
    plt.show()


# =============================================================================
# Losses distributed as a Pareto Type I Random Variable
# =============================================================================

def _compute_premium_pareto_intermediates(t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta,
                                          D=None, d=None, Principle=None, process_type=None):

    if k <= 2:
        raise ValueError("The k parameter has to be larger than 2")
    _validate_C_and_deductible(C, D, d)

    if d is not None:
        if not (0 < d <= 1):
            raise ValueError("d has to be in the interval (0, 1].")
        if d < (k - 1) / k:
            raise ValueError(f"d should be at least {(k - 1) / k} for this setting of k and eta_0.")
        D = d * k * eta_0 / (k - 1)
    else:
        if D < eta_0:
            raise ValueError(f"D (Deductible) has to be larger or equal than {eta_0} for this setting of k and eta_0.")

    exp_Nt = Expectation_Nt(t, alpha, lambda_0, xi, beta, process_type=process_type)
    var_Nt = Var_Nt(t, alpha, lambda_0, xi, beta, process_type=process_type)
    S = GLI(H_0, v, a, b)
    A = (S * eta_0**k) / ((k - 1) * D**(k - 1))

    loading = (2 * A * D / (k - 2) - A**2) * exp_Nt + var_Nt * A**2

    Pi, sqrt_or_loading = _apply_premium_principle(A, exp_Nt, loading, C, theta, Principle)
    return Pi, A, exp_Nt, var_Nt, S, sqrt_or_loading


def calculate_premium_pareto(t, k, eta_0, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                             D=None, d=None, Principle=None, process_type=None):

    Pi, _, _, _, _, _ = _compute_premium_pareto_intermediates(
        t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type
    )
    return Pi


def elasticity_premium_eta_0_pareto(t, k, eta_0, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                             D=None, d=None, Principle=None, process_type=None):

    Pi, A, exp_Nt, var_Nt, _, sqrt_or_loading = _compute_premium_pareto_intermediates(
        t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type
    )

    if Pi == 0:
        return 0.0

    # Case D:   dA/d_eta0 = k * A / eta_0  => factor k
    # Case d:   dA/d_eta0 = A / eta_0      => factor 1
    scale_factor = k if D is not None else 1.0

    if Principle == 'SDP':
        sqrt_val = sqrt_or_loading

        if d is not None:
            term_exp = (2 * theta * d * k * eta_0 / (sqrt_val * (k - 1) * (k - 2)) - (theta * A / sqrt_val) + 1) * exp_Nt
            term_var = (theta * A / sqrt_val) * var_Nt
            # From this particular case we expect to have elasticity 1.0 for all pair of parameters (k,eta_0)
            # (See that the relation of eta_0 with Pi is linear)
        else:
            term_exp = (theta * D / (sqrt_val * (k - 2)) - (theta * A / sqrt_val) + 1) * exp_Nt
            term_var = (theta * A / sqrt_val) * var_Nt

        epsilon_eta_0 = (C * A * scale_factor / Pi) * (term_exp + term_var)
        return float(epsilon_eta_0)

    elif Principle == 'VP':
        if d is not None:
            term_exp = (4 * theta * C * d * k * eta_0 / ((k - 1) * (k - 2)) - 2 * theta * C * A + 1) * exp_Nt
            term_var = 2 * theta * C * A * var_Nt
        else:
            term_exp = (2 * theta * C * D / (k - 2) - 2 * theta * C * A + 1) * exp_Nt
            term_var = 2 * theta * C * A * var_Nt

        epsilon_eta_0 = (C * A * scale_factor / Pi) * (term_exp + term_var)
        return float(epsilon_eta_0)

    else:
        raise ValueError("Principle must be either 'SDP' or 'VP'.")


def elasticity_premium_k_pareto(t, k, eta_0, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                D=None, d=None, Principle=None, process_type=None):

    Pi, A, exp_Nt, var_Nt, _, sqrt_or_loading = _compute_premium_pareto_intermediates(
        t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type
    )

    if Pi == 0:
        return 0.0

    if d is not None:
        log_term = np.log((k - 1.0) / (d * k)) - 1.0 / (k * (k - 1.0))    # Case d dA/dk
    else:
        log_term = np.log(eta_0 / D) - 1.0 / (k - 1.0)                    # Case D dA/dk

    if Principle == 'SDP':
        sqrt_val = sqrt_or_loading

        if d is not None:
            term_exp = (
                log_term * ((theta * d * k * eta_0) / (sqrt_val * (k - 1.0) * (k - 2.0)) - (theta * A) / sqrt_val + 1.0)
                - (theta * d * eta_0 * (k**2 - 2.0)) / (sqrt_val * ((k - 1.0)**2) * ((k - 2.0)**2))
            ) * exp_Nt
            term_var = log_term * (theta * A / sqrt_val) * var_Nt
        else:
            term_exp = (
                log_term * ((theta * D) / (sqrt_val * (k - 2.0)) - (theta * A) / sqrt_val + 1.0)
                - (theta * D) / (sqrt_val * ((k - 2.0)**2))
            ) * exp_Nt
            term_var = log_term * (theta * A / sqrt_val) * var_Nt

        epsilon_k = (C * A * k / Pi) * (term_exp + term_var)
        return float(epsilon_k)

    elif Principle == 'VP':
        if d is not None:
            term_exp = (
                log_term * ((2.0 * theta * C * d * k * eta_0) / ((k - 1.0) * (k - 2.0)) - 2.0 * theta * C * A + 1.0)
                - (2.0 * theta * C * d * eta_0 * (k**2 - 2.0)) / (((k - 2.0)**2) * ((k - 1.0)**2))
            ) * exp_Nt
            term_var = 2.0 * theta * C * A * log_term * var_Nt
        else:
            term_exp = (
                log_term * ((2.0 * theta * C * D) / (k - 2.0) - 2.0 * theta * C * A + 1.0)
                - (2.0 * theta * C * D) / ((k - 2.0)**2)
            ) * exp_Nt
            term_var = 2.0 * theta * C * A * log_term * var_Nt

        epsilon_k = (C * A * k / Pi) * (term_exp + term_var)
        return float(epsilon_k)

    else:
        raise ValueError("Principle must be either 'SDP' or 'VP'.")


# =============================================================================
# Pareto Type I -- 3D visualizations (same _plot_3d_param_surface skeleton
# used by the live Lomax/Lognormal Visual_* functions in plotting.py)
# =============================================================================

def _pareto_domain(D, d):
    """Domain used by the Pareto Visual_* functions -- k must stay away from
    the pole at (k-1)/k = d, and eta_0 (or the range implied by D) must stay
    inside the support eta_0 <= D."""
    max_k = min(1 / (1 - d) - 1e-5, 5) if d is not None else 5
    max_eta_0 = 50 if D is None else min(D - 1e-5, 50)
    return max_k, max_eta_0


def Visual_premium_pareto(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                          D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    max_k, max_eta_0 = _pareto_domain(D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        calculate_premium_pareto, t, np.linspace(2.075, max_k, 100), np.linspace(5, max_eta_0, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$k$", r"$\eta_0$", r"$\Pi$ (Premium)",
        f"Premium Calculation — {process_type} Process — Pareto Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_elasticity_premium_eta_0_pareto(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                    D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    max_k, max_eta_0 = _pareto_domain(D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        elasticity_premium_eta_0_pareto, t, np.linspace(2.075, max_k, 100), np.linspace(5, max_eta_0, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$k$", r"$\eta_0$", r"$\epsilon_{\eta_0}^{\Pi}$",
        f"Sensitivity Analysis: Elasticity w.r.t $\\eta_0$ — {process_type} Process — Pareto Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_elasticity_premium_k_pareto(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                               D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    condition_d = (1 / (1 - d) - 1e-5) if d is not None else 5
    condition_D = (D - 1e-5) if D is not None else 50
    max_k, max_eta_0 = min(condition_d, 5), min(condition_D, 50)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        elasticity_premium_k_pareto, t, np.linspace(2.075, max_k, 100), np.linspace(5, max_eta_0, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$k$", r"$\eta_0$", r"$\epsilon_{k}^{\Pi}$",
        f"Sensitivity Analysis: Elasticity w.r.t $k$ — {process_type} Process — Pareto Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


# =============================================================================
# Poisson-only insurance-decision framework (superseded by the Hawkes-vs-
# Poisson comparison plots actually used in the notebook: plot_value_
# function_vs_poisson / _plot_hawkes_vs_poisson_grid, in plotting.py)
# =============================================================================

def Initial_Net_Value_Poisson(
    V, h_grid, T_final, dist_type,
    k=None, eta_0=None, mu=None, sigma=None, C=None,
    theta=0.3, v=0.65, a=0.1, b=1.0, alpha=27.0, lambda_0=27, xi=15.0, beta=9.0,
    D=None, d=None, Principle=None
):
    init_v = V[:, -1].copy()
    dist_type = dist_type.lower()

    if dist_type == 'lomax':
        if k is None or eta_0 is None:
            raise ValueError("Parameters 'k' and 'eta_0' are required for Lomax.")
        calc_premium = lambda h_val: calculate_premium_lomax(
            t=T_final, k=k, eta_0=eta_0, C=C, theta=theta, v=v, a=a, b=b,
            alpha=alpha, xi=xi, beta=beta, H_0=h_val, lambda_0=lambda_0,
            D=D, d=d, Principle=Principle, process_type='poisson'
        )

    elif dist_type == 'lognormal':
        if mu is None or sigma is None:
            raise ValueError("Parameters 'mu' and 'sigma' are required for Lognormal.")
        calc_premium = lambda h_val: calculate_premium_lognormal(
            t=T_final, mu=mu, sigma=sigma, C=C, theta=theta, v=v, a=a, b=b,
            alpha=alpha, xi=xi, beta=beta, H_0=h_val, lambda_0=lambda_0,
            D=D, d=d, Principle=Principle, process_type='poisson'
        )

    else:
        raise ValueError(f"Unsupported distribution '{dist_type}'. Choose 'lomax' or 'lognormal'.")

    for j, h_val in enumerate(h_grid):
        init_v[j] -= calc_premium(h_val)

    return init_v


def initial_comparison_1_0_Poisson(
    dist_type, k=None, eta_0=None, mu=None, sigma=None, theta=0.3,
    t_0=0, T_final=1.0, K_steps=100, h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_0=27.0, rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=None, Principle=None,
    V1=None, V0=None, sub_Pre=True, save_names=None,
):
    _validate_tensor_arg(V1, "V1")
    _validate_tensor_arg(V0, "V0")

    h_grid = np.arange(h_min, h_max + 0.1, d_h)

    if V1 is None:
        V1, _ = Poisson_PDE_solver(
            dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=1,
            lambda_0=lambda_0, t_0=t_0, T_final=T_final, K_steps=K_steps,
            h_min=h_min, h_max=h_max, d_h=d_h, rho=rho, gamma=gamma, v=v, a=a, b=b, D=D, d=d
        )

    if sub_Pre:
        init_V1 = Initial_Net_Value_Poisson(
            V1, h_grid=h_grid, T_final=T_final, dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma,
            C=1, theta=theta, v=v, a=a, b=b, alpha=alpha, lambda_0=lambda_0, xi=xi, beta=beta, D=D, d=d, Principle=Principle
        )
    else:
        init_V1 = V1[:, -1]

    if V0 is None:
        V0, _ = Poisson_PDE_solver(
            dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=0,
            lambda_0=lambda_0, t_0=t_0, T_final=T_final, K_steps=K_steps,
            h_min=h_min, h_max=h_max, d_h=d_h, rho=rho, gamma=gamma, v=v, a=a, b=b, D=D, d=d
        )

    init_V0 = V0[:, -1]

    V_max_t0 = np.maximum(init_V1, init_V0)
    mask_t0 = init_V1 > init_V0

    from scipy.interpolate import interp1d
    interp_v1 = interp1d(h_grid, init_V1, kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_v0 = interp1d(h_grid, init_V0, kind='cubic', bounds_error=False, fill_value="extrapolate")

    h_fine = np.linspace(h_min, h_max, 1000)
    v1_fine = interp_v1(h_fine)
    v0_fine = interp_v0(h_fine)

    matriz1_fine = np.maximum(v1_fine, v0_fine)
    mask_V1_fine = v1_fine > v0_fine

    dist_lower = str(dist_type).lower()
    if dist_lower in ['lomax', 'lomax_ii']:
        param_str = f"Lomax ($k={k if k is not None else a}$, $\\eta_0={eta_0 if eta_0 is not None else b}$)"
    else:
        param_str = f"Log-Normal ($\\mu={mu}$, $\\sigma={sigma}$)"

    principle_str = f" | Premium Principle: {Principle}" if Principle else ""
    theta_str = f" ($\\theta={theta}$)" if theta is not None else ""
    lambda_str = f" ($\\lambda_0={lambda_0}$)"
    title_text = (
        r"$\bf{Optimal\ Cyber-Insurance\ Decision\ Framework\ at\ Initial\ Time\ (t=0)}$" + "\n" +
        f"Poisson Model{lambda_str} | {param_str}{principle_str}{theta_str}"
    )

    hex_v0, hex_v1 = "#D62728", "#1F77B4"
    cmap_binary = ListedColormap([hex_v0, hex_v1])

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.8, 0.82],
                            wspace=0.15, hspace=0.35, top=0.88, bottom=0.08, left=0.06, right=0.95)
    fig.suptitle(title_text, fontsize=13, x=0.5, y=0.95, ha='center', va='center')

    legend_elements = [
        Line2D([0], [0], color=hex_v1, lw=4, label=r"Insurance: $V_1(0,h) > V_0(0,h$)"),
        Line2D([0], [0], color=hex_v0, lw=4, label=r"No Insurance: $V_0(0,h) \geq V_1(0,h$)")
    ]

    ax1 = fig.add_subplot(gs[0, 0])
    v1_cut = np.where(mask_V1_fine, matriz1_fine, np.nan)
    v0_cut = np.where(~mask_V1_fine, matriz1_fine, np.nan)
    ax1.plot(h_fine, v1_cut, color=hex_v1, lw=2.5)
    ax1.plot(h_fine, v0_cut, color=hex_v0, lw=2.5)
    ax1.set_xlabel(r"$h$", fontsize=10)
    ax1.set_ylabel(r"$V(0,h)$", fontsize=10)
    ax1.set_title(r"$V(0,h) = \max{[V_1(0,h),V_0(0,h)]}$", fontsize=11, pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pcolormesh(h_fine, np.array([0, 1]), np.tile(mask_V1_fine.astype(int), (2, 1)),
                   cmap=cmap_binary, shading="nearest", alpha=0.9, vmin=0, vmax=1)
    ax2.set_xlabel(r"$h$", fontsize=10, labelpad=25)
    ax2.set_yticks([])
    ax2.set_title(r"Decision Regions at $t=0$ in $h$ Space", fontsize=11, pad=12)
    ax2.set_box_aspect(0.2)
    ax2.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.65),
               ncol=1, framealpha=0.85, fontsize=8)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(h_grid, init_V1, color=hex_v1, lw=2.0, linestyle="-", label=r"$V_1(0,h)$ (Insurance)")
    ax3.plot(h_grid, init_V0, color=hex_v0, lw=2.0, linestyle="--", label=r"$V_0(0,h)$ (No Insurance)")
    ax3.set_xlabel(r"$h$", fontsize=10)
    ax3.set_ylabel(r"$V(0,h)$", fontsize=10)
    ax3.set_title(r"Comparison $V_1(0,h)$ vs. $V_0(0,h)$", fontsize=12, pad=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)

    _save_panels(fig, [ax1, ax2, ax3], save_names)
    plt.show()
    return V_max_t0, mask_t0, V1, init_V1, V0, init_V0


def compare_Hawkes_vs_Poisson_ht(
    V_hawkes, V_poisson, h_min=0.0, h_max=50.0, t_0=0.0, T_final=1.0,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0, lambda_val=27.0,
    dist_type="Lomax", Principle="SDP", theta=0.03, reverse_t=True,
    save_name=None,
):
    if isinstance(V_hawkes, tuple):
        V_hawkes = V_hawkes[0]
    if isinstance(V_poisson, tuple):
        V_poisson = V_poisson[0]

    n_lambda, n_h_hkw, n_t_hkw = V_hawkes.shape
    n_h_poi, n_t_poi = V_poisson.shape

    h_grid = np.linspace(h_min, h_max, n_h_poi)
    t_grid = np.linspace(t_0, T_final, n_t_poi)
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)

    if n_lambda == len(lambda_grid):
        idx_lambda = np.argmin(np.abs(lambda_grid - lambda_val))
        V_hawkes_2d = V_hawkes[idx_lambda, :, :]
        used_lambda = lambda_grid[idx_lambda]
    else:
        V_hawkes_2d = V_hawkes[0, :, :]
        used_lambda = lambda_val

    V_poisson_plot = V_poisson[:, ::-1] if reverse_t else V_poisson
    V_hawkes_plot = V_hawkes_2d[:, ::-1] if reverse_t else V_hawkes_2d

    H_mesh, T_mesh = np.meshgrid(h_grid, t_grid)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    v_min_global = min(V_hawkes_plot.min(), V_poisson_plot.min())
    v_max_global = max(V_hawkes_plot.max(), V_poisson_plot.max())

    surf_hkw = ax.plot_surface(T_mesh, H_mesh, V_hawkes_plot.T, cmap="Blues", alpha=0.75,
                                edgecolor="none", vmin=v_min_global, vmax=v_max_global)
    surf_poi = ax.plot_surface(T_mesh, H_mesh, V_poisson_plot.T, cmap="YlOrRd", alpha=0.65,
                                edgecolor="none", vmin=v_min_global, vmax=v_max_global)

    ax.view_init(elev=25, azim=-60)
    ax.set_xlabel(r"$t$", labelpad=8, fontsize=10)
    ax.set_ylabel(r"$h$", labelpad=8, fontsize=10)
    ax.set_zlabel(r"$V(t, h)$", labelpad=8, fontsize=10)
    ax.set_title(
        f"Model Comparison: Hawkes vs. Poisson Process\n"
        f"Distribution: {dist_type} | Premium Principle: {Principle} ($\\theta={theta}$)\n"
        f"Value Functions $V(t, h)$ — Hawkes ($\\lambda={used_lambda:.1f}$) vs. Poisson",
        fontsize=11, pad=15
    )
    cbar_hkw = fig.colorbar(surf_hkw, ax=ax, shrink=0.5, pad=0.06, aspect=12)
    cbar_hkw.set_label(r"Hawkes $V_{\text{Hawkes}}(t,h)$", fontsize=9)
    cbar_poi = fig.colorbar(surf_poi, ax=ax, shrink=0.5, pad=0.14, aspect=12)
    cbar_poi.set_label(r"Poisson $V_{\text{Poisson}}(t,h)$", fontsize=9)

    _save_figure(fig, save_name)
    plt.tight_layout()
    plt.show()
