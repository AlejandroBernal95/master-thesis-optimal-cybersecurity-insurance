"""
All visualization utilities for the thesis: process trajectories, premium
surfaces and elasticity sensitivity, value-function / optimal-control fields,
and insurance-decision maps (risk-neutral and risk-averse). Consolidated here
so every plotting function lives in one place; the domain modules
(processes.py, premiums.py, pide_risk_neutral.py, pide_risk_averse.py) keep
only the numerical/solver logic.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import differential_evolution
from scipy.integrate import quad


from processes import Expectation_Nt, Var_Nt, Simulate_Process, lambda_t_computation, Expectation_Lambda_t
from premiums import (
    GLI, GLII, calculate_premium_lomax, calculate_premium_lognormal,
    elasticity_premium_eta_0_lomax, elasticity_premium_k_lomax,
    elasticity_premium_mu, elasticity_premium_sigma,
    _validate_C_and_deductible,
)
from pide_risk_neutral import (
    Hawkes_PIDE_solver, Poisson_PDE_solver, Initial_Net_Value_Hawkes, z_star, z_star_Poisson,
)
from validation import compute_static_trajectory_for_pair, evaluate_J_Hawkes_constant_policy
from pide_risk_averse import solve_W_zbar_RA, Hawkes_PIDE_solver_RA, Initial_Net_Value_Hawkes_RA

import os

THESIS_PLOTS_DIR = "thesis_plots"


def _save_figure(fig, name, output_dir=THESIS_PLOTS_DIR, dpi=300):
    """Saves the WHOLE figure as one file -- for single-panel plots."""
    if name is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def _save_panels(fig, axes_list, names, output_dir=THESIS_PLOTS_DIR, dpi=300):
    """Saves each axes in axes_list as its OWN individual file, named from
    `names` (must be the same length as axes_list) -- for multi-panel plots.
    Works for 2D and 3D axes; uses each axes' own tight bounding box so only
    that panel (not the whole composite figure) ends up in each file."""
    if names is None:
        return
    if len(names) != len(axes_list):
        raise ValueError(f"Expected {len(axes_list)} names, got {len(names)}.")
    os.makedirs(output_dir, exist_ok=True)
    fig.canvas.draw()
    for ax, name in zip(axes_list, names):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, bbox_inches=extent, dpi=dpi)
        print(f"Saved: {path}")


# =============================================================================
# Process trajectories (processes.py)
# =============================================================================

def Visual_Process(process_type, T_final=1, alpha=27, lambda_0=27, xi=15, beta=9, save_names=None):
    """N_t vs. E[N_t] for either process; also plots lambda_t vs. E[lambda_t]
    for Hawkes (the Poisson intensity is constant and not informative to plot)."""

    process_type = process_type.lower()
    x = np.linspace(0, T_final, 500)
    events = Simulate_Process(process_type, T_final, alpha, lambda_0, xi, beta)

    exp_Nt = Expectation_Nt(x, alpha, lambda_0, xi, beta, process_type=process_type)
    Process_N = [len([y for y in events if y <= i]) for i in x]

    if process_type == 'hawkes':
        exp_Lt = Expectation_Lambda_t(x, alpha, lambda_0, xi, beta, process_type=process_type)
        Process_L = [lambda_t_computation(i, alpha, lambda_0, xi, beta, events) for i in x]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(x, Process_N, color='blue', linewidth=1.5, label='$N_t$')
        ax1.plot(x, exp_Nt, color='grey', linestyle='--', label='$E[N_t]$')
        ax1.set_ylabel('$N_t$')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2.plot(x, Process_L, color='blue', linewidth=1.5, label='$\\lambda_t$')
        ax2.plot(x, exp_Lt, color='grey', linestyle='--', label='$E[\\lambda_t]$')
        ax2.set_ylabel('$\\lambda_t$')
        ax2.set_xlabel('$t$ (years)')
        ax2.legend(loc='upper left')
        ax2.grid(True)

    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(x, Process_N, color='blue', linewidth=1.5, label='$N_t$')
        ax1.plot(x, exp_Nt, color='grey', linestyle='--', label='$E[N_t]$')
        ax1.set_ylabel('$N_t$')
        ax1.set_xlabel('$t$ (years)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

    if process_type == 'hawkes':
        _save_panels(fig, [ax1, ax2], save_names)
    else:
        _save_panels(fig, [ax1], save_names)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Security Breach Probability Functions (Gordon-Loeb)
# =============================================================================

def SBPF(class_type, v, a, b = None, save_name = None):

    x = np.linspace(0, 200, 1000)
    
    if class_type == 1:
        if b is None:
            raise ValueError("Parameter 'b' is required when class_type is 1.")
        y = GLI(x, v, a, b)
        ylabel = r"$S_{I}$"
    elif class_type == 2:
        y = GLII(x, v, a)
        ylabel = r"$S_{II}$"
    else:
        raise ValueError("Parameter 'class_type' only admits 1 or 2 as possible values.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color="blue", linewidth=1.5, label=f"Class {class_type}")
    ax.set_xlabel(r"z")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    _save_figure(fig, save_name)
    plt.show()

# =============================================================================
# Premium surfaces and elasticity sensitivity (premiums.py)
# =============================================================================

def _format_ded_principle_str(D, d, Principle):
    ded_str = f"Fixed Deductible (D = {D})" if D is not None else f"Proportional Deductible (d = {d})"
    principle_str = "Variance Principle (VP)" if Principle == "VP" else "Standard Deviation Principle (SDP)"
    return ded_str, principle_str


def _plot_3d_param_surface(func, t, x_vals, y_vals, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta,
                            D, d, Principle, process_type, x_label, y_label, z_label, title, save_name=None):

    X, Y = np.meshgrid(x_vals, y_vals)
    calc_vectorized = np.vectorize(func)
    Z = calc_vectorized(t, X, Y, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    _save_figure(fig, save_name)
    plt.show()


def Visual_premium_lomax(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                          D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        calculate_premium_lomax, t, np.linspace(2.075, 5, 100), np.linspace(5, 25, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$k$", r"$\eta_0$", r"$\Pi$ (Premium)",
        f"Premium Calculation — {process_type} Process — Lomax Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_elasticity_premium_eta_0_lomax(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                           D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        elasticity_premium_eta_0_lomax, t, np.linspace(2.075, 5, 100), np.linspace(5, 25, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$k$", r"$\eta_0$", r"$\epsilon_{\eta_0}^{\Pi}$",
        f"Sensitivity Analysis: Elasticity w.r.t $\\eta_0$ — {process_type} Process — Lomax Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_elasticity_premium_k_lomax(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                      D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        elasticity_premium_k_lomax, t, np.linspace(2.075, 5, 100), np.linspace(5, 25, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$k$", r"$\eta_0$", r"$\epsilon_{k}^{\Pi}$",
        f"Sensitivity Analysis: Elasticity w.r.t $k$ — {process_type} Process — Lomax Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_premium_lognormal(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                               D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        calculate_premium_lognormal, t, np.linspace(0.75, 2.0, 100), np.linspace(0.5, 1.5, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$\mu$", r"$\sigma$", r"$\Pi$ (Premium)",
        f"Premium Calculation — {process_type} Process — Lognormal Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_elasticity_premium_mu(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        elasticity_premium_mu, t, np.linspace(0.75, 2.0, 100), np.linspace(0.5, 1.5, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$\mu$", r"$\sigma$", r"$\epsilon_{\mu}^{\Pi}$",
        f"Sensitivity Analysis: Elasticity w.r.t $\\mu$ — {process_type} Process — Lognormal Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )


def Visual_elasticity_premium_sigma(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                   D=None, d=None, Principle=None, process_type=None, save_name=None):
    _validate_C_and_deductible(C, D, d)
    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    _plot_3d_param_surface(
        elasticity_premium_sigma, t, np.linspace(0.75, 2.0, 100), np.linspace(0.5, 1.5, 100),
        C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        r"$\mu$", r"$\sigma$", r"$\epsilon_{\sigma}^{\Pi}$",
        f"Sensitivity Analysis: Elasticity w.r.t $\\sigma$ — {process_type} Process — Lognormal Losses\n{principle_str} — {ded_str}",
        save_name=save_name,
    )

# =============================================================================
# Hawkes vs. Poisson premium comparison
# =============================================================================

def truncate_colormap(cmap_name, minval=0.25, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap_name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def Premium_comparison(t, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1.0, alpha=27.0,
                       lambda_0=27.0, lambda_poisson=None, xi=15.0, beta=9.0, D=None, d=None, Principle=None, dist_type=None,
                       save_name=None):
    """Compares premium surfaces (Hawkes vs. Poisson) in a single 3D plot.
    Supports Lomax and Lognormal distributions.

    lambda_0: Hawkes initial intensity, used for the Hawkes surface.
    lambda_poisson: constant Poisson rate used for the Poisson surface.
    Defaults to lambda_0 if not given (unchanged behavior)."""

    _validate_C_and_deductible(C, D, d)
    dist_lower = dist_type.lower()
    lam_p = lambda_0 if lambda_poisson is None else lambda_poisson

    if dist_lower == 'lomax':
        x_vals = np.linspace(2.075, 5.0, 60)
        y_vals = np.linspace(5.0, 25.0, 60)
        x_label, y_label = r"$k$", r"$\eta_0$"
        calc_func = np.vectorize(calculate_premium_lomax)

    elif dist_lower == 'lognormal':
        x_vals = np.linspace(0.1, 2.0, 60)
        y_vals = np.linspace(0.1, 1.5, 60)
        x_label, y_label = r"$\mu$", r"$\sigma$"
        calc_func = np.vectorize(calculate_premium_lognormal)

    else:
        raise ValueError(f"Unsupported distribution type '{dist_type}'. Choose 'lomax' or 'lognormal'.")

    X, Y = np.meshgrid(x_vals, y_vals)

    Z_hawkes = calc_func(t, X, Y, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type="Hawkes")
    Z_poisson = calc_func(t, X, Y, C, theta, H_0, v, a, b, alpha, lam_p, xi, beta, D, d, Principle, process_type="Poisson")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm_h = PowerNorm(gamma=0.25, vmin=Z_hawkes.min(), vmax=Z_hawkes.max())
    norm_p = PowerNorm(gamma=0.25, vmin=Z_poisson.min(), vmax=Z_poisson.max())

    
    blues_trunc = truncate_colormap('Blues', 0.25, 1.0)
    ylorrd_trunc = truncate_colormap('YlOrRd', 0.25, 1.0)
    
    surf_hawkes = ax.plot_surface(X, Y, Z_hawkes, cmap=blues_trunc, norm=norm_h, alpha=0.9, edgecolor='none')
    surf_poisson = ax.plot_surface(X, Y, Z_poisson, cmap=ylorrd_trunc, norm=norm_p, alpha=0.85, edgecolor='none')

    cbar_hawkes = fig.colorbar(surf_hawkes, ax=ax, shrink=0.5, aspect=10, pad=0.05)
    cbar_hawkes.set_label(r"$\Pi_{\text{Hawkes}}$", fontsize=10)

    cbar_poisson = fig.colorbar(surf_poisson, ax=ax, shrink=0.5, aspect=10, pad=0.08)
    cbar_poisson.set_label(r"$\Pi_{\text{Poisson}}$", fontsize=10)

    ded_str, principle_str = _format_ded_principle_str(D, d, Principle)
    dist_str = "Lomax" if dist_lower == 'lomax' else "Lognormal"

    ax.set_title(
        f"Premium Comparison: Hawkes vs. Poisson Process ({dist_str} Losses)\n{principle_str} — {ded_str}",
        fontsize=12, pad=15
    )
    ax.set_xlabel(x_label, fontsize=11, labelpad=8)
    ax.set_ylabel(y_label, fontsize=11, labelpad=8)
    ax.set_zlabel(r"$\Pi$ (Premium)", fontsize=11, labelpad=8)

    _save_figure(fig, save_name)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Value function / optimal control fields, insurance decision maps (RN)
# (pide_risk_neutral.py)
# =============================================================================

def _dist_param_str(dist_type, k, eta_0, mu, sigma):
    dist_lower = str(dist_type).lower()
    if dist_lower in ["lomax", "lomax_ii"]:
        return f"Lomax ($k={k}$, $\\eta_0={eta_0}$)"
    return f"Lognormal ($\\mu={mu}$, $\\sigma={sigma}$)"


def _plot_field_over_t_lambda_h(
    field, field_label, main_title,
    lambda_fixed, h_fixed, t_0, T_final,
    h_min, h_max, d_h, lambda_min, lambda_max, d_lambda,
    surface_color="#3182bd", line_color="#1f77b4",
    lambda_plot_min=None, lambda_plot_max=None,
    save_names=None,
):
    """Shared 2x2 layout (3D field vs. (t,h) | 3D field vs. (t,lambda) |
    2D trajectory over t | blank) used by both plot_value_function and
    plot_optimal_control -- the two were structurally identical in the
    original notebook, differing only in which tensor they plot and the
    axis/label text.

    lambda_plot_min/lambda_plot_max restrict the DISPLAYED range of the
    field-vs-(t,lambda) surface (subplot 2) only -- lambda_grid is still
    built from the real (lambda_min, lambda_max, d_lambda) solve grid to
    correctly index into `field`, which was solved on that full grid. This
    is purely cosmetic: it does not affect lambda_fixed (subplot 1, 3) or
    the underlying tensor."""

    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"Expected a numpy array for 'field', got {type(field).__name__}. "
            "Every solver in this project (Hawkes_PIDE_solver, Poisson_PDE_solver, "
            "Hawkes_PIDE_solver_RA, ...) returns a (tensor, meta) tuple -- did you "
            "forget to unpack it? e.g. `V, _ = Hawkes_PIDE_solver(...)` instead of "
            "`V = Hawkes_PIDE_solver(...)`."
        )

    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)

    n_t = field.shape[2] if field.ndim == 3 else 1
    t_eval = np.linspace(T_final, t_0, n_t)

    idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))
    idx_h = np.argmin(np.abs(h_grid - h_fixed))
    val_lambda_actual = lambda_grid[idx_lambda]
    val_h_actual = h_grid[idx_h]

    plot_lo = lambda_min if lambda_plot_min is None else lambda_plot_min
    plot_hi = lambda_max if lambda_plot_max is None else lambda_plot_max
    lambda_display_mask = (lambda_grid >= plot_lo) & (lambda_grid <= plot_hi)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.20, hspace=0.25,
                            left=0.06, right=0.94, top=0.90, bottom=0.08)
    fig.suptitle(main_title, fontsize=12, x=0.5, y=0.97, ha="center")

    # Subplot 1: field(t, h) for fixed lambda
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    grid_1 = field[idx_lambda, :, :]
    mesh_x1, mesh_y1 = np.meshgrid(t_eval, h_grid)
    ax1.plot_surface(mesh_x1, mesh_y1, grid_1, color=surface_color, alpha=0.85, edgecolor="none", shade=True)
    ax1.set_xlabel(r"$t$", labelpad=6, fontsize=11)
    ax1.set_ylabel(r"$h$", labelpad=6, fontsize=11)
    ax1.set_zlabel(field_label, labelpad=6, fontsize=11)
    ax1.view_init(elev=25, azim=-55)
    ax1.plot([], [], [], color=surface_color, label=f"$\\lambda = {val_lambda_actual:.0f}$")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Subplot 2: field(t, lambda) for fixed h -- lambda axis restricted to
    # [plot_lo, plot_hi] to avoid displaying the numerically unstable tail.
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    grid_2 = field[lambda_display_mask, idx_h, :]
    mesh_x2, mesh_y2 = np.meshgrid(t_eval, lambda_grid[lambda_display_mask])
    ax2.plot_surface(mesh_x2, mesh_y2, grid_2, color=surface_color, alpha=0.85, edgecolor="none", shade=True)
    ax2.set_xlabel(r"$t$", labelpad=6, fontsize=11)
    ax2.set_ylabel(r"$\lambda$", labelpad=6, fontsize=11)
    ax2.set_zlabel(field_label, labelpad=6, fontsize=11)
    ax2.view_init(elev=25, azim=-55)
    ax2.plot([], [], [], color=surface_color, label=f"$h = {val_h_actual:.0f}$")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Subplot 3: trajectory over t for fixed (lambda, h)
    ax3 = fig.add_subplot(gs[1, 0])
    trajectory = field[idx_lambda, idx_h, :]
    ax3.plot(t_eval, trajectory, color=line_color, linewidth=1.8,
              label=f"$\\lambda = {val_lambda_actual:.0f}, \\ h = {val_h_actual:.0f}$")
    ax3.set_xlabel(r"$t$", fontsize=11)
    ax3.set_ylabel(field_label, fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=10)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    _save_panels(fig, [ax1, ax2, ax3], save_names)
    plt.show()


_CONTEXT_INFO = {
    "RN": dict(symbol=r"$V$", name="Value\\ Function\\ V"),
    "RA": dict(symbol=r"$W$", name="Certainty\\ Equivalent\\ W"),
}


def plot_value_function(
    V, context, lambda_fixed=27.0, h_fixed=0.0, t_0=0.0, T_final=1.0,
    h_min=0.0, h_max=50.0, d_h=0.5, lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    dist_type="Lomax", k=None, eta_0=None, mu=None, sigma=None, C=None, d=None,
    lambda_plot_min=27.0, lambda_plot_max=150.0,
    save_names=None,
):
    """context: 'RN' (risk-neutral -- V is a Value Function) or 'RA'
    (risk-averse -- V is a Certainty Equivalent W). Required, no default:
    the caller must state which one V actually is, since this function has
    no way to infer it from the tensor itself."""
    context = context.upper()
    if context not in _CONTEXT_INFO:
        raise ValueError(f"context must be 'RN' or 'RA', got '{context}'.")
    info = _CONTEXT_INFO[context]

    param_str = _dist_param_str(dist_type, k, eta_0, mu, sigma)

    if C == 0:
        title = r"$\bf{" + info["name"] + r"(t, \lambda, h)}$" + "\n" + f"Hawkes Model | {param_str} | C = {C}"
    else:
        title = r"$\bf{" + info["name"] + r"(t, \lambda, h)}$" + "\n" + f"Hawkes Model | {param_str} | C = {C} | d = {d}"

    _plot_field_over_t_lambda_h(
        V, info["symbol"], title, lambda_fixed, h_fixed, t_0, T_final,
        h_min, h_max, d_h, lambda_min, lambda_max, d_lambda,
        lambda_plot_min=lambda_plot_min, lambda_plot_max=lambda_plot_max,
        save_names=save_names,
    )



def plot_optimal_control(
    V, gamma=0.05, lambda_fixed=27.0, h_fixed=0.0, t_0=0.0, T_final=1.0,
    h_min=0.0, h_max=50.0, d_h=0.5, lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    dist_type="Lomax", k=None, eta_0=None, mu=None, sigma=None, C=None, d=None,
    lambda_plot_min=27.0, lambda_plot_max=150.0,
    save_names=None,
):
    z_star_matrix = z_star(V, d_lambda, d_h, gamma)
    param_str = _dist_param_str(dist_type, k, eta_0, mu, sigma)
    
    if C == 0:
        title = r"$\bf{Optimal\ Control\ z^*(t, \lambda, h)}$" + "\n" + f"Hawkes Model | {param_str} | C = {C}"
    else:
        title = r"$\bf{Optimal\ Control\ z^*(t, \lambda, h)}$" + "\n" + f"Hawkes Model | {param_str} | C = {C} | d = {d}"
        
    _plot_field_over_t_lambda_h(
        z_star_matrix, r"$z^*$", title, lambda_fixed, h_fixed, t_0, T_final,
        h_min, h_max, d_h, lambda_min, lambda_max, d_lambda,
        line_color="#08519c",
        lambda_plot_min=lambda_plot_min, lambda_plot_max=lambda_plot_max,
        save_names=save_names,
    )
    return z_star_matrix


_SENSITIVITY_LABELS = {
    "d": "d", "rho": r"\rho", "xi": r"\xi", "gamma": r"\gamma", "alpha": r"\alpha",
    "beta": r"\beta", "k": "k", "eta_0": r"\eta_0", "C": "C", "theta": r"\theta", "omega": r"\omega"
}


def plot_parameter_sensitivity(
    param_name, param_grid, lambda_fixed, h_fixed,
    dist_type=None, k=None, eta_0=None, mu=None, sigma=None, C=None,
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=None,
    save_names=None,
):
    """Sweeps a single Hawkes_PIDE_solver parameter (param_name) over
    param_grid, solving V once per value, and plots V(t) and z*(t) at a
    fixed (lambda, h) side by side. Generalizes the three near-identical
    d/rho/xi sensitivity functions into one -- works for any solver
    parameter, not just those three."""
    if not (lambda_min <= lambda_fixed <= lambda_max):
        raise ValueError("lambda_fixed has to be in the interval [lambda_min, lambda_max]")
    if not (h_min <= h_fixed <= h_max):
        raise ValueError("h_fixed has to be in the interval [h_min, h_max]")

    solver_kwargs = dict(
        dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=C,
        t_0=t_0, T_final=T_final, K_steps=K_steps,
        h_min=h_min, h_max=h_max, d_h=d_h,
        lambda_min=lambda_min, lambda_max=lambda_max, d_lambda=d_lambda,
        rho=rho, gamma=gamma, v=v, a=a, b=b,
        alpha=alpha, xi=xi, beta=beta, D=D, d=d, U_sqrt = True,
    )
    if param_name not in solver_kwargs:
        raise ValueError(f"'{param_name}' is not a Hawkes_PIDE_solver parameter that can be swept.")

    V_by_val, z_by_val = {}, {}
    for value in param_grid:
        print(f"Solving V for {param_name}={value} ...")
        solver_kwargs[param_name] = value
        V_tensor, (N, M, t_eval, lambda_grid, h_grid) = Hawkes_PIDE_solver(**solver_kwargs)

        z_opt = z_star(V=V_tensor, d_lambda=solver_kwargs["d_lambda"], d_h=solver_kwargs["d_h"], gamma=solver_kwargs["gamma"])
        idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))
        idx_h = np.argmin(np.abs(h_grid - h_fixed))

        V_by_val[value] = V_tensor[idx_lambda, idx_h, :]
        z_by_val[value] = z_opt[idx_lambda, idx_h, :]

    symbol = _SENSITIVITY_LABELS.get(param_name, param_name)
    param_str = _dist_param_str(dist_type, k, eta_0, mu, sigma)

    base_sub = f"Hawkes Model | {param_str} | C = {C}"
    if C != 0 and param_name != "d" and d is not None:
        base_sub += f" | d = {d}"
    base_sub += f" | $\\lambda={lambda_fixed}$, $h={h_fixed}$"

    title_V = r"$\bf{Value\ Function\ V(t, \lambda, h)\ Sensitivity\ to\ " + symbol + r"}$" + "\n" + base_sub
    title_z = r"$\bf{Optimal\ Control\ z^*(t, \lambda, h)\ Sensitivity\ to\ " + symbol + r"}$" + "\n" + base_sub

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    for value in param_grid:
        ax1.plot(t_eval, np.squeeze(V_by_val[value]), label=f"${symbol} = {value}$")
        ax2.plot(t_eval, np.squeeze(z_by_val[value]), label=f"${symbol} = {value}$")

    #ax1.set_title(title_V, fontsize=10, pad=10)
    ax1.set_xlabel(r"Time $t$", fontsize=10)
    #ax1.set_ylabel(r"$V(t)$", fontsize=10)
    ax1.set_ylabel(r"$W(t)$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(title=f"Parameter ${symbol}$", loc="best")

    #ax2.set_title(title_z, fontsize=10, pad=10)
    ax2.set_xlabel(r"Time $t$", fontsize=10)
    ax2.set_ylabel(r"$z^*(t)$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(title=f"Parameter ${symbol}$", loc="best")

    _save_panels(fig, [ax1, ax2], save_names)
    plt.tight_layout()
    return fig, (ax1, ax2)


# =============================================================================
# Insurance decision maps (V1 = C=1 vs. V0 = C=0) and Hawkes-vs-Poisson plots
# =============================================================================


def _validate_tensor_arg(x, name):
    """Raises a clear error instead of a cryptic downstream AttributeError
    when a caller forgets to unpack a solver's (tensor, meta) return."""
    if x is not None and not isinstance(x, np.ndarray):
        raise TypeError(
            f"Expected a numpy array for '{name}', got {type(x).__name__}. "
            "Every solver in this project (Hawkes_PIDE_solver, Poisson_PDE_solver, "
            "Hawkes_PIDE_solver_RA, ...) returns a (tensor, meta) tuple -- did you "
            f"forget to unpack it? e.g. `{name}, _ = Hawkes_PIDE_solver(...)` instead "
            f"of `{name} = Hawkes_PIDE_solver(...)`."
        )


_PANEL_TITLES_1_0 = {
    "value_surface": r"$V(0,\lambda,h) = \max{[V_1(0,\lambda,h),V_0(0,\lambda,h)]}$",
    "decision_map": r"Decision Boundaries at $t=0$ in $(h, \lambda)$ Space",
    "comparison_surface": r"Comparison $V_1(0,\lambda,h)$ vs. $V_0(0,\lambda,h)$",
}


def initial_comparison_1_0_Hawkes(
    dist_type, k=None, eta_0=None, mu=None, sigma=None, theta=0.3,
    t_0=0, T_final=1.0, K_steps=100, h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0, rho=0.2, gamma=0.05,
    v=0.65, a=0.1, b=1.0, alpha=27.0, xi=15.0, beta=9.0, D=None, d=None,
    Principle=None, V1=None, V0=None, sub_Pre=True,
    lambda_plot_min=27.0, lambda_plot_max=150.0,
    panel=None, show_suptitle=True, show_panel_title=False,
    save_names=None,
):
    """
    panel: None (default) draws the full layout -- value-surface cut |
    decision map | comparison surface | blank -- exactly as before. Pass
    one of 'value_surface', 'decision_map', 'comparison_surface' to draw
    ONLY that panel as its own standalone figure.
    show_suptitle: toggles the overall figure title (model / distribution /
    premium principle).
    show_panel_title: toggles each individual panel's own title. Independent
    of show_suptitle -- e.g. for a thesis figure with its own \\caption{},
    you likely want both off (the current thesis default: both were already
    off before this switch existed).
    """
    if panel is not None and panel not in _PANEL_TITLES_1_0:
        raise ValueError(f"panel must be one of {list(_PANEL_TITLES_1_0)} or None, got '{panel}'.")

    _validate_tensor_arg(V1, "V1")
    _validate_tensor_arg(V0, "V0")

    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)

    if V1 is None:
        V1, _ = Hawkes_PIDE_solver(
            dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=1,
            t_0=t_0, T_final=T_final, K_steps=K_steps, h_min=h_min, h_max=h_max, d_h=d_h,
            lambda_min=lambda_min, lambda_max=lambda_max, d_lambda=d_lambda,
            rho=rho, gamma=gamma, v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta, D=D, d=d
        )

    if sub_Pre:
        init_V1 = Initial_Net_Value_Hawkes(
            V1, lambda_grid=lambda_grid, h_grid=h_grid, T_final=T_final,
            dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=1, theta=theta,
            v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta, D=D, d=d, Principle=Principle,
        )
    else:
        init_V1 = V1[:, :, -1]

    if V0 is None:
        V0, _ = Hawkes_PIDE_solver(
            dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=0,
            t_0=t_0, T_final=T_final, K_steps=K_steps, h_min=h_min, h_max=h_max, d_h=d_h,
            lambda_min=lambda_min, lambda_max=lambda_max, d_lambda=d_lambda,
            rho=rho, gamma=gamma, v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta, D=D, d=d
        )

    init_V0 = V0[:, :, -1]

    V_max_t0 = np.maximum(init_V1, init_V0)
    mask_t0 = init_V1 > init_V0

    interp_v1 = RegularGridInterpolator((lambda_grid, h_grid), init_V1, bounds_error=False, fill_value=None)
    interp_v0 = RegularGridInterpolator((lambda_grid, h_grid), init_V0, bounds_error=False, fill_value=None)

    h_fine = np.linspace(h_min, h_max, 1000)
    lambda_fine = np.linspace(lambda_min, lambda_max, 1000)
    L_fine, H_fine = np.meshgrid(lambda_fine, h_fine, indexing='ij')

    pts_fine = np.column_stack([L_fine.ravel(), H_fine.ravel()])
    v1_fine = interp_v1(pts_fine).reshape(L_fine.shape)
    v0_fine = interp_v0(pts_fine).reshape(L_fine.shape)

    matriz1_fine = np.maximum(v1_fine, v0_fine)
    mask_V1_fine = v1_fine > v0_fine

    plot_lo = lambda_min if lambda_plot_min is None else lambda_plot_min
    plot_hi = lambda_max if lambda_plot_max is None else lambda_plot_max
    fine_mask = (lambda_fine >= plot_lo) & (lambda_fine <= plot_hi)
    coarse_mask = (lambda_grid >= plot_lo) & (lambda_grid <= plot_hi)

    L_fine_disp = L_fine[fine_mask, :]
    H_fine_disp = H_fine[fine_mask, :]
    matriz1_fine_disp = matriz1_fine[fine_mask, :]
    mask_V1_fine_disp = mask_V1_fine[fine_mask, :]

    lambda_grid_disp = lambda_grid[coarse_mask]
    init_V1_disp = init_V1[coarse_mask, :]
    init_V0_disp = init_V0[coarse_mask, :]

    dist_lower = str(dist_type).lower()
    if dist_lower in ['lomax', 'lomax_ii']:
        param_str = f"Lomax ($k={k if k is not None else a}$, $\\eta_0={eta_0 if eta_0 is not None else b}$)"
    else:
        param_str = f"Log-Normal ($\\mu={mu}$, $\\sigma={sigma}$)"

    principle_str = f" | Premium Principle: {Principle}" if Principle else ""
    theta_str = f" ($\\theta={theta}$)" if theta is not None else ""
    title_text = (
        r"$\bf{Optimal\ Cyber-Insurance\ Decision\ Framework\ at\ Initial\ Time\ (t=0)}$" + "\n" +
        f"Hawkes Model | {param_str}{principle_str}{theta_str}"
    )

    hex_v0, hex_v1 = "#D62728", "#1F77B4"
    cmap_binary = ListedColormap([hex_v0, hex_v1])

    # NOTE: the second legend entry used to read "$C = 0.9$" -- this panel only
    # ever compares C=1 vs C=0 (see the two Hawkes_PIDE_solver calls above), so
    # that was a mislabel. Fixed to "$C = 0$" here.
    legend_elements = [
        Line2D([0], [0], color=hex_v1, lw=4, label=r"$C = 1$"),
        Line2D([0], [0], color=hex_v0, lw=4, label=r"$C = 0$"),
    ]

    def _draw_value_surface(ax):
        v1_cut = np.where(mask_V1_fine_disp, matriz1_fine_disp, np.nan)
        v0_cut = np.where(~mask_V1_fine_disp, matriz1_fine_disp, np.nan)
        ax.plot_surface(H_fine_disp.T, L_fine_disp.T, v1_cut.T, color=hex_v1, alpha=0.95, shade=False, edgecolor="none")
        ax.plot_surface(H_fine_disp.T, L_fine_disp.T, v0_cut.T, color=hex_v0, alpha=0.95, shade=False, edgecolor="none")
        ax.set_xlabel(r"$h$", labelpad=8)
        ax.set_ylabel(r"$\lambda$", labelpad=8)
        ax.set_zlabel(r"$V(0,\lambda,h)$", labelpad=8)
        if show_panel_title:
            ax.set_title(_PANEL_TITLES_1_0["value_surface"], fontsize=11, pad=12)

    def _draw_decision_map(ax):
        ax.pcolormesh(H_fine_disp.T, L_fine_disp.T, mask_V1_fine_disp.T.astype(int), cmap=cmap_binary,
                       shading="nearest", alpha=0.9, vmin=0, vmax=1)
        ax.set_xlabel(r"$h$", fontsize=10)
        ax.set_ylabel(r"$\lambda$", fontsize=10)
        ax.set_box_aspect(0.85)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=1, framealpha=0.9, fontsize=8, borderaxespad=0.0)
        if show_panel_title:
            ax.set_title(_PANEL_TITLES_1_0["decision_map"], fontsize=11, pad=12)

    def _draw_comparison_surface(ax):
        L_mesh, H_mesh = np.meshgrid(lambda_grid_disp, h_grid, indexing='ij')
        # NOTE: surf1 used to be normalized to its OWN local min/max while
        # surf0 used the global (v_min_global, v_max_global) -- inconsistent
        # color mapping between the two surfaces on the same z-axis. Both now
        # share the global normalization (the commented-out line already in
        # the original code showed this was the intended behavior).
        v_min_global = min(init_V1_disp.min(), init_V0_disp.min())
        v_max_global = max(init_V1_disp.max(), init_V0_disp.max())
        ax.plot_surface(H_mesh.T, L_mesh.T, init_V1_disp.T, cmap="Blues", alpha=1, edgecolor="none",
                         vmin=v_min_global, vmax=v_max_global)
        ax.plot_surface(H_mesh.T, L_mesh.T, init_V0_disp.T, cmap="YlOrRd", alpha=0.7, edgecolor="none",
                         vmin=v_min_global, vmax=v_max_global)
        ax.set_xlabel(r"$h$", labelpad=8)
        ax.set_ylabel(r"$\lambda$", labelpad=8)
        # NOTE: this used to be mislabeled "$z^*$" -- this panel plots
        # init_V1_disp/init_V0_disp (value surfaces), not the optimal control.
        ax.set_zlabel(r"$V(0,\lambda,h)$", labelpad=6)
        ax.legend(handles=legend_elements, framealpha=0.9, fontsize=8)
        if show_panel_title:
            ax.set_title(_PANEL_TITLES_1_0["comparison_surface"], fontsize=12, pad=10)

    draw_fn = {
        "value_surface": _draw_value_surface,
        "decision_map": _draw_decision_map,
        "comparison_surface": _draw_comparison_surface,
    }

    if panel is not None:
        is_3d = panel in ("value_surface", "comparison_surface")
        fig = plt.figure(figsize=(9, 7) if is_3d else (7, 6))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
        if show_suptitle:
            fig.suptitle(title_text, fontsize=13, x=0.5, y=0.97, ha='center', va='center')
        draw_fn[panel](ax)
        _save_panels(fig, [ax], save_names)
        plt.tight_layout()
        plt.show()
        return V_max_t0, mask_t0, V1, init_V1, V0, init_V0

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.2], width_ratios=[1.8, 0.82],
                        wspace=0.12, hspace=0.35, top=0.88, bottom=0.05, left=0.05, right=0.95)
    if show_suptitle:
        fig.suptitle(title_text, fontsize=13, x=0.5, y=0.95, ha='center', va='center')

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_value_surface(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_decision_map(ax2)

    ax3 = fig.add_subplot(gs[1, :], projection="3d")
    _draw_comparison_surface(ax3)

    _save_panels(fig, [ax1, ax2, ax3], save_names)
    plt.show()
    return V_max_t0, mask_t0, V1, init_V1, V0, init_V0


# =============================================================================
# Optimal coinsurance decision map (risk-averse), swept over a C_grid
# (pide_risk_averse.py)
# =============================================================================

def plot_optimal_C_decision_map_RA(
    W_sweep, C_grid, theta=0.3, Principle=None,
    n_fine=1000, save_name=None,
    show_frontier_boundary=False, cmap_name='turbo',
    show_region_labels=False,
    lambda_plot_min=None, lambda_plot_max=None,
):
    """
    Part 2 of the optimal-C decision map: takes the output of
    solve_W_by_C_RA (W(0,lambda,h;C) already solved for each C -- no PIDE
    solve happens here) plus a premium configuration (theta, Principle),
    computes A(C;lambda,h) = W(0,lambda,h;C) - Pi(C;lambda,h) at each grid
    point, picks the argmax, and renders the decision map. Cheap enough to
    call repeatedly while sweeping theta, reusing the same W_sweep.

    C_grid must be a subset of the C values already solved for in
    W_sweep -- raises if any are missing (re-run solve_W_by_C_RA with a
    wider C_grid to fix).

    show_frontier_boundary: set False to suppress the red "not
    identifiable" contour/legend entry -- useful when C_grid = {0, 1} and
    an optimum below/above the tested range genuinely doesn't apply.
    cmap_name: colormap for the region fill. 'turbo' (default) spreads
    adjacent categories across very different hues, which reads much more
    clearly than 'viridis' once C_grid has more than 3-4 levels close
    together (viridis's darker end compresses adjacent bands visually).
    show_region_labels: overlay the C value as text directly on each
    region (placed at the region's own centroid, falling back to the
    nearest in-region point if the centroid itself isn't in the mask --
    e.g. for a crescent-shaped or disconnected region), so reading the
    map doesn't depend on matching colors to the colorbar at all.
    lambda_plot_min/lambda_plot_max: restrict the DISPLAYED lambda range
    (same cosmetic-only convention as plot_value_function/plot_optimal_control
    -- avoids showing the numerically unstable tail). Defaults to the full
    (lambda_min, lambda_max) already solved for in W_sweep if not given.
    """
    W_by_C = W_sweep['W_by_C']
    lambda_grid = W_sweep['lambda_grid']
    h_grid = W_sweep['h_grid']
    ctx = W_sweep['context']

    missing = [C for C in C_grid if C not in W_by_C]
    if missing:
        raise ValueError(
            f"C_grid contains values not present in W_sweep: {missing}. "
            "Re-run solve_W_by_C_RA with a C_grid that includes them."
        )

    n_C = len(C_grid)

    # --- A(C;lambda,h) = W(0,lambda,h;C) - Pi(C;lambda,h) ---
    A_by_C = {}
    for C in C_grid:
        A_by_C[C] = Initial_Net_Value_Hawkes_RA(
            W=W_by_C[C], lambda_grid=lambda_grid, h_grid=h_grid,
            T_final=ctx['T_final'], dist_type=ctx['dist_type'],
            k=ctx['k'], eta_0=ctx['eta_0'], mu=ctx['mu'], sigma=ctx['sigma'],
            C=C, theta=theta, v=ctx['v'], a=ctx['a'], b=ctx['b'],
            alpha=ctx['alpha'], xi=ctx['xi'], beta=ctx['beta'],
            D=ctx['D'], d=ctx['d'], Principle=Principle,
        )

    A_stack = np.stack([A_by_C[C] for C in C_grid], axis=0)  # (n_C, N, M)
    idx_best = np.argmax(A_stack, axis=0)  # (N, M) -- index into C_grid

    frontier_mask = (idx_best == 0)

    # --- interpolate onto a fine mesh for smooth decision boundaries ---
    h_min, h_max = h_grid[0], h_grid[-1]
    lambda_min, lambda_max = lambda_grid[0], lambda_grid[-1]
    plot_lo = lambda_min if lambda_plot_min is None else lambda_plot_min
    plot_hi = lambda_max if lambda_plot_max is None else lambda_plot_max
    h_fine = np.linspace(h_min, h_max, n_fine)
    lambda_fine = np.linspace(plot_lo, plot_hi, n_fine)
    L_fine, H_fine = np.meshgrid(lambda_fine, h_fine, indexing='ij')
    pts_fine = np.column_stack([L_fine.ravel(), H_fine.ravel()])

    A_fine_stack = np.zeros((n_C, n_fine, n_fine))
    for idx_C, C in enumerate(C_grid):
        interp_C = RegularGridInterpolator((lambda_grid, h_grid), A_by_C[C], bounds_error=False, fill_value=None)
        A_fine_stack[idx_C] = interp_C(pts_fine).reshape(L_fine.shape)

    idx_best_fine = np.argmax(A_fine_stack, axis=0)
    frontier_mask_fine = (idx_best_fine == 0)

    # --- decision map ---
    dist_lower = str(ctx['dist_type']).lower()
    if dist_lower in ['lomax', 'lomax_ii']:
        param_str = f"Lomax ($k={ctx['k']}$, $\\eta_0={ctx['eta_0']}$)"
    else:
        param_str = f"Log-Normal ($\\mu={ctx['mu']}$, $\\sigma={ctx['sigma']}$)"

    cmap = plt.get_cmap(cmap_name, n_C)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(h_fine, lambda_fine, idx_best_fine, cmap=cmap,
                        vmin=-0.5, vmax=n_C - 0.5, shading='auto')

    cbar = fig.colorbar(im, ax=ax, ticks=range(n_C))
    cbar.ax.set_yticklabels([f"C = {C}" for C in C_grid])
    cbar.set_label(r"Optimal $C$")

    # Thin lines between EVERY band (not just the frontier one), so band
    # edges are visually crisp even when adjacent colors are close.
    all_level_edges = np.arange(-0.5, n_C - 0.5 + 1e-9, 1.0)
    ax.contour(h_fine, lambda_fine, idx_best_fine, levels=all_level_edges,
               colors='white', linewidths=0.6, alpha=0.8)

    if show_region_labels:
        for idx_C, C in enumerate(C_grid):
            mask = (idx_best_fine == idx_C)
            if not mask.any():
                continue
            i_idx, j_idx = np.where(mask)
            # geometric centroid in (lambda, h) coordinates
            lam_c = lambda_fine[i_idx].mean()
            h_c = h_fine[j_idx].mean()
            # snap to the closest point that is actually IN the mask, in
            # case the region is crescent-/ring-shaped and its centroid
            # falls outside it (e.g. Image 1's lens-shaped C=0.9 band)
            lam_pts, h_pts = lambda_fine[i_idx], h_fine[j_idx]
            dist2 = (lam_pts - lam_c) ** 2 + (h_pts - h_c) ** 2
            k_nearest = np.argmin(dist2)
            lam_label, h_label = lam_pts[k_nearest], h_pts[k_nearest]
            ax.text(h_label, lam_label, f"{C:g}", ha='center', va='center',
                     fontsize=9, fontweight='bold', color='white',
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='black')])

    if show_frontier_boundary:
        ax.contour(h_fine, lambda_fine, frontier_mask_fine.astype(int), levels=[0.5],
                   colors='red', linewidths=2.5)
        ax.plot([], [], color='red', linewidth=2.5,
                label=f"Boundary: $C^* \\leq {C_grid[0]}$ (not identifiable, below tested range)")
        ax.legend(loc='upper right', fontsize=9)

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(
        r"$\bf{Optimal\ Coinsurance\ C^*(\lambda,h)\ at\ }t=0$" + "\n" +
        f"Hawkes Model | {param_str} | $r={ctx['r']:g}$ | $\\theta={theta}$"
    )
    _save_figure(fig, save_name)
    plt.tight_layout()
    plt.show()

    return idx_best, frontier_mask, A_by_C


# =============================================================================
# Dynamic (PIDE) vs. static-control gain (pide_risk_neutral.py)
# =============================================================================

def plot_dynamic_vs_static_gain(
    V_tensor, dist_type, k=None, eta_0=None, mu=None, sigma=None, C=0,
    lambda_fixed=27.0, h_targets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=0.1, z_bounds=(0.0, 50.0),
    save_name=None,
):
    """
    Percentage gain of dynamic control over static control (thesis eq. 22):
        gain(t) = 100 * (V_dynamic(t,lambda,h) - J_static(t,lambda,h)) / J_static(t,lambda,h)
    V_dynamic is read off V_tensor (solved backward, T -> 0, by
    Hawkes_PIDE_solver); J_static is recomputed at each t via
    compute_static_trajectory_for_pair, one call per h in h_targets.
    """
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    t_eval_forward = np.linspace(t_0, T_final, K_steps)

    _validate_tensor_arg(V_tensor, "V_tensor")
    idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))
    V_dynamic_forward = V_tensor[:, :, ::-1]  # backward-solved (T->0) -> forward (0->T)

    fig = plt.figure(figsize=(8, 5))
    for h_val in h_targets:
        idx_h = np.argmin(np.abs(h_grid - h_val))
        v_dynamic = V_dynamic_forward[idx_lambda, idx_h, :]

        j_static = compute_static_trajectory_for_pair(
            l_val=lambda_fixed, h_val=h_val, dist_type=dist_type, k=k, eta_0=eta_0,
            mu=mu, sigma=sigma, C=C, t_eval=t_eval_forward, T_final=T_final,
            rho=rho, gamma=gamma, v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta,
            D=D, d=d, z_bounds=z_bounds,
        )

        gain = 100.0 * (v_dynamic - j_static) / j_static
        plt.plot(t_eval_forward, gain, label=f"$\\lambda={lambda_fixed:.0f}, h={h_val}$", linewidth=1.8)

    plt.xlabel("$t$")
    plt.ylabel("% gain")
    plt.title("Dynamic vs. Static Control: Value Gain Over Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    _save_figure(fig, save_name)
    plt.tight_layout()
    plt.show()


def plot_dynamic_vs_constant_gain(
    V_tensor, z_val, dist_type, k=None, eta_0=None, mu=None, sigma=None, C=0,
    lambda_fixed=27.0, h_targets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=0.1,
    save_name=None,
):
    """
    Percentage gain of the PIDE's dynamic control over a FIXED CONSTANT
    control z_val held over [t,T]:
        gain(t) = 100 * (V_dynamic(t,lambda,h) - J(t,lambda,h;z_val)) / J(t,lambda,h;z_val)
    Same structure as plot_dynamic_vs_static_gain, but z_val is given
    directly (e.g. z_val=0, "never invest") instead of being found by
    differential_evolution at each t -- so this shows how costly a
    particular fixed policy is relative to the dynamic optimum, not just
    the best possible constant.
    """
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    t_eval_forward = np.linspace(t_0, T_final, K_steps)

    _validate_tensor_arg(V_tensor, "V_tensor")
    idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))
    V_dynamic_forward = V_tensor[:, :, ::-1]  # backward-solved (T->0) -> forward (0->T)

    fig = plt.figure(figsize=(8, 5))
    for h_val in h_targets:
        idx_h = np.argmin(np.abs(h_grid - h_val))
        v_dynamic = V_dynamic_forward[idx_lambda, idx_h, :]

        j_const = evaluate_J_Hawkes_constant_policy(
            z_val=z_val, l_val=lambda_fixed, h_val=h_val, dist_type=dist_type, k=k, eta_0=eta_0,
            mu=mu, sigma=sigma, C=C, t_eval=t_eval_forward, T_final=T_final,
            rho=rho, gamma=gamma, v=v, a=a, b=b, alpha=alpha, xi=xi, beta=beta,
            D=D, d=d,
        )

        gain = 100.0 * (v_dynamic - j_const) / j_const
        plt.plot(t_eval_forward, gain, label=f"$\\lambda={lambda_fixed:.0f}, h={h_val}$", linewidth=1.8)

    plt.xlabel("$t$")
    plt.ylabel("% gain")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    _save_figure(fig, save_name)
    plt.tight_layout()
    plt.show()

def _plot_hawkes_vs_poisson_grid(
    field_hawkes, field_poisson, field_label, main_title,
    lambda_fixed, h_fixed, t_fixed, t_0, T_final,
    h_min, h_max, d_h, lambda_min, lambda_max, d_lambda,
    hawkes_color="#3182bd", poisson_color="#e6550d",
    lambda_plot_min=None, lambda_plot_max=None,
    save_names=None,
):
    """Shared 2x2 layout (one blank cell) comparing a Hawkes field(t,lambda,h)
    against its Poisson counterpart (no lambda-dependence, broadcast where
    needed). lambda_plot_min/lambda_plot_max restrict the DISPLAYED lambda
    range of panel B only (the panel with an explicit lambda axis) -- same
    cosmetic-only restriction as plot_value_function/plot_optimal_control."""

    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    n_t = field_hawkes.shape[2]
    t_eval = np.linspace(T_final, t_0, n_t)

    idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))
    idx_h = np.argmin(np.abs(h_grid - h_fixed))
    idx_t = np.argmin(np.abs(t_eval - t_fixed))
    val_lambda_actual = lambda_grid[idx_lambda]
    val_h_actual = h_grid[idx_h]
    val_t_actual = t_eval[idx_t]

    plot_lo = lambda_min if lambda_plot_min is None else lambda_plot_min
    plot_hi = lambda_max if lambda_plot_max is None else lambda_plot_max
    lambda_display_mask = (lambda_grid >= plot_lo) & (lambda_grid <= plot_hi)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.20, hspace=0.25,
                            left=0.06, right=0.94, top=0.90, bottom=0.08)
    fig.suptitle(main_title, fontsize=12, x=0.5, y=0.97, ha="center")

    axA = fig.add_subplot(gs[0, 0], projection="3d")
    mesh_t, mesh_h = np.meshgrid(t_eval, h_grid)
    axA.plot_surface(mesh_t, mesh_h, field_hawkes[idx_lambda, :, :],
                      color=hawkes_color, alpha=0.75, edgecolor="none", shade=True)
    axA.plot_surface(mesh_t, mesh_h, field_poisson[:, :],
                      color=poisson_color, alpha=0.75, edgecolor="none", shade=True)
    axA.set_xlabel(r"$t$", labelpad=6, fontsize=11)
    axA.set_ylabel(r"$h$", labelpad=6, fontsize=11)
    axA.set_zlabel(field_label, labelpad=6, fontsize=11)
    axA.view_init(elev=25, azim=-55)
    axA.plot([], [], [], color=hawkes_color, label=f"Hawkes, $\\lambda={val_lambda_actual:.0f}$")
    axA.plot([], [], [], color=poisson_color, label=f"Poisson, $\\lambda^P={val_lambda_actual:.0f}$")
    axA.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel B: field(h, lambda) at t_fixed -- lambda axis restricted to
    # [plot_lo, plot_hi] to avoid displaying the numerically unstable tail.
    axB = fig.add_subplot(gs[0, 1], projection="3d")
    lambda_grid_disp = lambda_grid[lambda_display_mask]
    mesh_h2, mesh_lambda2 = np.meshgrid(h_grid, lambda_grid_disp)
    Z_hawkes_B = field_hawkes[lambda_display_mask, :, idx_t]
    Z_poisson_B = np.tile(field_poisson[:, idx_t], (len(lambda_grid_disp), 1))
    axB.plot_surface(mesh_h2, mesh_lambda2, Z_hawkes_B, color=hawkes_color, alpha=0.75, edgecolor="none", shade=True)
    axB.plot_surface(mesh_h2, mesh_lambda2, Z_poisson_B, color=poisson_color, alpha=0.75, edgecolor="none", shade=True)
    axB.set_xlabel(r"$h$", labelpad=6, fontsize=11)
    axB.set_ylabel(r"$\lambda$", labelpad=6, fontsize=11)
    axB.set_zlabel(field_label, labelpad=6, fontsize=11)
    axB.view_init(elev=25, azim=-55)
    axB.plot([], [], [], color=hawkes_color, label=f"Hawkes, $t={val_t_actual:.2g}$")
    axB.plot([], [], [], color=poisson_color, label=f"Poisson, $\\lambda^P={val_lambda_actual:.0f}$, $t={val_t_actual:.2g}$")
    axB.legend(loc="upper right", framealpha=0.9, fontsize=9)

    axC = fig.add_subplot(gs[1, 0])
    axC.plot(t_eval, field_hawkes[idx_lambda, idx_h, :], color=hawkes_color, linewidth=1.8,
             label=f"Hawkes, $\\lambda={val_lambda_actual:.0f}$, $h={val_h_actual:.0f}$")
    axC.plot(t_eval, field_poisson[idx_h, :], color=poisson_color, linewidth=1.8, linestyle="--",
             label=f"Poisson, $\\lambda^P={val_lambda_actual:.0f}$, $h={val_h_actual:.0f}$")
    axC.set_xlabel(r"$t$", fontsize=11)
    axC.set_ylabel(field_label, fontsize=11)
    axC.grid(True, linestyle="--", alpha=0.5)
    axC.legend(loc="best", framealpha=0.9, fontsize=9)

    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")

    _save_panels(fig, [axA, axB, axC], save_names)
    plt.show()

def plot_value_function_vs_poisson(
    V_hawkes, J_poisson, context, lambda_fixed=27.0, h_fixed=0.0, t_fixed=0.0, t_0=0.0, T_final=1.0,
    h_min=0.0, h_max=50.0, d_h=0.5, lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    dist_type="Lomax", k=None, eta_0=None, mu=None, sigma=None, C=None, d=None,
    lambda_plot_min=27.0, lambda_plot_max=150.0,
    hawkes_color="#3182bd", poisson_color="#e6550d",
    save_names=None,
):
    """context: 'RN' or 'RA', same meaning as in plot_value_function --
    required for the same reason (this pair is called with both RN and RA
    tensors in the notebook, so there is no safe default)."""
    context = context.upper()
    if context not in _CONTEXT_INFO:
        raise ValueError(f"context must be 'RN' or 'RA', got '{context}'.")
    info = _CONTEXT_INFO[context]

    param_str = f"Lomax ($k={k}$, $\\eta_0={eta_0}$)" if str(dist_type).lower().startswith("lomax") else f"Lognormal ($\\mu={mu}$, $\\sigma={sigma}$)"

    if C == 0:
        title = r"$\bf{" + info["name"] + r"(t, \lambda, h)}$" + "\n" + f"Hawkes vs. Poisson | {param_str} | C = {C}"
    else:
        title = r"$\bf{" + info["name"] + r"(t, \lambda, h)}$" + "\n" + f"Hawkes vs. Poisson | {param_str} | C = {C} | d = {d}"

    _plot_hawkes_vs_poisson_grid(
        V_hawkes, J_poisson, info["symbol"], title,
        lambda_fixed, h_fixed, t_fixed, t_0, T_final,
        h_min, h_max, d_h, lambda_min, lambda_max, d_lambda,
        hawkes_color=hawkes_color, poisson_color=poisson_color,
        lambda_plot_min=lambda_plot_min, lambda_plot_max=lambda_plot_max,
        save_names=save_names,
    )

def plot_optimal_control_vs_poisson(
    V_hawkes, J_poisson, gamma=0.05,
    lambda_fixed=27.0, h_fixed=0.0, t_fixed=0.0, t_0=0.0, T_final=1.0,
    h_min=0.0, h_max=50.0, d_h=0.5, lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    dist_type="Lomax", k=None, eta_0=None, mu=None, sigma=None, C=None, d=None,
    lambda_plot_min=27.0, lambda_plot_max=150.0,
    hawkes_color="#08519c", poisson_color="#a63603",
    save_names=None,
):
    """Computes z*(t,lambda,h) (Hawkes) and z*(t,h) (Poisson) internally from
    V_hawkes / V_poisson -- same pattern as plot_optimal_control, which
    computes z_star from V rather than requiring the caller to pre-compute
    it."""
    z_hawkes = z_star(V_hawkes, d_lambda, d_h, gamma)
    z_poisson = z_star_Poisson(J_poisson, d_h, gamma)

    param_str = f"Lomax ($k={k}$, $\\eta_0={eta_0}$)" if str(dist_type).lower().startswith("lomax") else f"Lognormal ($\\mu={mu}$, $\\sigma={sigma}$)"

    
    if C == 0:
        title = r"$\bf{Optimal\ Control\ z^*(t, \lambda, h)}$" + "\n" + f"Hawkes vs. Poisson | {param_str} | C = {C}"
    else:
        title = r"$\bf{Optimal\ Control\ z^*(t, \lambda, h)}$" + "\n" + f"Hawkes vs. Poisson | {param_str} | C = {C} | d = {d}"
        
    _plot_hawkes_vs_poisson_grid(
        z_hawkes, z_poisson, r"$z^*$", title,
        lambda_fixed, h_fixed, t_fixed, t_0, T_final,
        h_min, h_max, d_h, lambda_min, lambda_max, d_lambda,
        hawkes_color=hawkes_color, poisson_color=poisson_color,
        lambda_plot_min=lambda_plot_min, lambda_plot_max=lambda_plot_max,
        save_names=save_names,
    )
    return z_hawkes, z_poisson

def plot_parameter_sensitivityRA(
    param_name, param_grid, lambda_fixed, h_fixed,
    dist_type=None, k=None, eta_0=None, mu=None, sigma=None, C=None,
    omega = 1, r = 50, t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    rho=0.2, gamma=0.05, v=0.65, a=0.1, b=1.0,
    alpha=27.0, xi=15.0, beta=9.0, D=None, d=None,
    save_names=None,
):
    """Sweeps a single Hawkes_PIDE_solver parameter (param_name) over
    param_grid, solving V once per value, and plots V(t) and z*(t) at a
    fixed (lambda, h) side by side. Generalizes the three near-identical
    d/rho/xi sensitivity functions into one -- works for any solver
    parameter, not just those three."""
    if not (lambda_min <= lambda_fixed <= lambda_max):
        raise ValueError("lambda_fixed has to be in the interval [lambda_min, lambda_max]")
    if not (h_min <= h_fixed <= h_max):
        raise ValueError("h_fixed has to be in the interval [h_min, h_max]")

    solver_kwargs = dict(
        dist_type=dist_type, k=k, eta_0=eta_0, mu=mu, sigma=sigma, C=C,
        omega = omega, r = r, t_0=t_0, T_final=T_final, K_steps=K_steps,
        h_min=h_min, h_max=h_max, d_h=d_h,
        lambda_min=lambda_min, lambda_max=lambda_max, d_lambda=d_lambda,
        rho=rho, gamma=gamma, v=v, a=a, b=b,
        alpha=alpha, xi=xi, beta=beta, D=D, d=d
    )
    if param_name not in solver_kwargs:
        raise ValueError(f"'{param_name}' is not a Hawkes_PIDE_solver parameter that can be swept.")

    V_by_val, z_by_val = {}, {}
    for value in param_grid:
        print(f"Solving V for {param_name}={value} ...")
        solver_kwargs[param_name] = value
        V_tensor, (N, M, t_eval, lambda_grid, h_grid) = Hawkes_PIDE_solver_RA(**solver_kwargs)

        z_opt = z_star(V=V_tensor, d_lambda=solver_kwargs["d_lambda"], d_h=solver_kwargs["d_h"], gamma=solver_kwargs["gamma"])
        idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))
        idx_h = np.argmin(np.abs(h_grid - h_fixed))

        V_by_val[value] = V_tensor[idx_lambda, idx_h, :]
        z_by_val[value] = z_opt[idx_lambda, idx_h, :]

    symbol = _SENSITIVITY_LABELS.get(param_name, param_name)
    param_str = _dist_param_str(dist_type, k, eta_0, mu, sigma)

    base_sub = f"Hawkes Model | {param_str} | C = {C}"
    if C != 0 and param_name != "d" and d is not None:
        base_sub += f" | d = {d}"
    base_sub += f" | $\\lambda={lambda_fixed}$, $h={h_fixed}$"

    title_V = r"$\bf{Value\ Function\ V(t, \lambda, h)\ Sensitivity\ to\ " + symbol + r"}$" + "\n" + base_sub
    title_z = r"$\bf{Optimal\ Control\ z^*(t, \lambda, h)\ Sensitivity\ to\ " + symbol + r"}$" + "\n" + base_sub

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    for value in param_grid:
        ax1.plot(t_eval, np.squeeze(V_by_val[value]), label=f"${symbol} = {value}$")
        ax2.plot(t_eval, np.squeeze(z_by_val[value]), label=f"${symbol} = {value}$")

    #ax1.set_title(title_V, fontsize=10, pad=10)
    ax1.set_xlabel(r"Time $t$", fontsize=10)
    ax1.set_ylabel(r"$W(t)$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(title=f"Parameter ${symbol}$", loc="best")

    #ax2.set_title(title_z, fontsize=10, pad=10)
    ax2.set_xlabel(r"Time $t$", fontsize=10)
    ax2.set_ylabel(r"$z^*(t)$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(title=f"Parameter ${symbol}$", loc="best")

    _save_panels(fig, [ax1, ax2], save_names)
    plt.tight_layout()
    return fig, (ax1, ax2)



############################################################################################################################

def compute_dynamic_vs_zbar_gain_RA(
    W_tensor, W_zbar_tensor, dist_type, k=None, eta_0=None, mu=None, sigma=None, C=0.8,
    lambda_fixed=27.0, h_targets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    omega =1, t_0=0.0, T_final=1.0, K_steps=100,
    h_min=0.0, h_max=50.0, d_h=0.5,
    lambda_min=27.0, lambda_max=216.0, d_lambda=1.0,
    save_name=None, plot=True,
):
    """
    Certainty-equivalent gain of the dynamic optimal control over a FIXED
    z_bar held constant over [t,T]:

        Gain(t,lambda,h) = W(t,lambda,h;C) - W_zbar(t,lambda,h;C,z_bar)

    in the SAME dollar units as W itself -- NOT a percentage, and NOT
    computed via V. V = -exp(-omega(x+W)) compresses toward numbers
    indistinguishable from 0 once X_0 is large (a defining property of
    CARA, not a bug), so a %gain built directly on V (or on raw exp'd
    objectives) would be pure cancellation noise -- see the earlier
    discussion in this thread. W is already the de-exponentiated,
    dollar-scale object; comparing W to W_zbar directly is the stable
    analog of the risk-neutral %gain, at the cost of losing the "%" framing
    (a dollar gain doesn't have a natural denominator to normalize by here
    the way J > 0 does in the risk-neutral case, since W_zbar can be
    negative).

    Unlike plot_dynamic_vs_constant_gain (risk-neutral), which computes
    J(t,lambda,h;z_val) on the fly via a cheap closed-form quad per point,
    this takes W_tensor and W_zbar_tensor as ALREADY-SOLVED PIDE outputs
    (from Hawkes_PIDE_solver_RA and solve_W_zbar_RA respectively) -- under
    CARA there's no cheap per-point evaluation, so both tensors must be
    solved on the full grid beforehand.

    W_tensor and W_zbar_tensor must have been solved on the SAME
    (lambda_grid, h_grid, t_eval) grid -- their shapes are checked, but a
    shape match doesn't guarantee the underlying grids (d_h, d_lambda,
    lambda_min/max, h_min/max) were actually identical, so pass the same
    grid kwargs to both solver calls that produced them.

    Returns (gains, t_eval_forward): gains is a dict {h_val: gain_array},
    each array indexed forward in time (t=0 -> T). Also plots unless
    plot=False.
    """
    lambda_grid = np.arange(lambda_min, lambda_max + 0.1, d_lambda)
    h_grid = np.arange(h_min, h_max + 0.1, d_h)
    t_eval_forward = np.linspace(t_0, T_final, K_steps)

    _validate_tensor_arg(W_tensor, "W_tensor")
    _validate_tensor_arg(W_zbar_tensor, "W_zbar_tensor")
    if W_tensor.shape != W_zbar_tensor.shape:
        raise ValueError(
            f"W_tensor shape {W_tensor.shape} != W_zbar_tensor shape {W_zbar_tensor.shape}. "
            "Both must be solved on the same (lambda_grid, h_grid, t_eval) grid -- "
            "check that Hawkes_PIDE_solver_RA and solve_W_zbar_RA were called with "
            "matching h_min/h_max/d_h, lambda_min/lambda_max/d_lambda, and K_steps."
        )

    idx_lambda = np.argmin(np.abs(lambda_grid - lambda_fixed))

    # both tensors are backward-solved (T -> 0); flip to forward (0 -> T)
    # to match plot_dynamic_vs_constant_gain's convention
    W_dynamic_forward = W_tensor[:, :, ::-1]
    W_zbar_forward = W_zbar_tensor[:, :, ::-1]

    gains = {}
    if plot:
        fig = plt.figure(figsize=(8, 5))
    for h_val in h_targets:
        idx_h = np.argmin(np.abs(h_grid - h_val))
        w_dynamic = W_dynamic_forward[idx_lambda, idx_h, :]
        w_zbar = W_zbar_forward[idx_lambda, idx_h, :]
        gain = (w_dynamic - w_zbar)
        gains[h_val] = gain
        if plot:
            plt.plot(t_eval_forward, gain, label=f"$\\lambda={lambda_fixed:.0f}, h={h_val}$", linewidth=1.8)

    if plot:
        plt.xlabel("$t$")
        plt.ylabel(r"Gain $= W - W_{\bar z}$")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        _save_figure(fig, save_name)
        plt.tight_layout()
        plt.show()

    return gains, t_eval_forward