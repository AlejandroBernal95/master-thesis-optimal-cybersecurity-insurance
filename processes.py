"""
Hawkes and Poisson point processes: statistical moments and Ogata's thinning
simulation. Visualization utilities (Visual_Process, Var_Nt_plot) live in
plotting.py.

Hawkes intensity (exponential kernel):
    lambda_t = alpha + (lambda_0 - alpha) * exp(-xi * t) + beta * sum_i exp(-xi*(t - t_i))
    alpha  > 0 : long-term mean intensity
    lambda_0 > 0 : intensity at t = 0
    xi     > 0 : exponential decay rate
    beta   > 0 : magnitude of self-excitation (requires xi > beta, subcriticality)
"""

import numpy as np


# =============================================================================
# Statistical Moments
# =============================================================================

def Expectation_Lambda_t(t, alpha=27, lambda_0=27, xi=15, beta=9, process_type='hawkes'):

    process_type = process_type.lower()

    if process_type == 'hawkes':
        xi_mb = xi - beta
        return alpha * xi / xi_mb + np.exp(-xi_mb * t) * (lambda_0 - alpha * xi / xi_mb)

    elif process_type == 'poisson':
        return np.full_like(np.asarray(t, dtype=float), lambda_0)

    else:
        raise ValueError(f"Unsupported jump process '{process_type}'. Choose 'hawkes' or 'poisson'.")

def Variance_Lambda_t(t, alpha=27, lambda_0=27, xi=15, beta=9, process_type='hawkes'):
    process_type = process_type.lower()
    if process_type == 'hawkes':
        xi_mb = xi - beta
        term_const = alpha * xi * beta**2 / (2 * xi_mb**2)
        term_exp1 = beta**2 * (lambda_0 * xi_mb - alpha * xi) / xi_mb**2 * np.exp(-xi_mb * t)
        term_exp2 = beta**2 * (alpha * xi - 2 * lambda_0 * xi_mb) / (2 * xi_mb**2) * np.exp(-2 * xi_mb * t)
        return term_const + term_exp1 + term_exp2
    elif process_type == 'poisson':
        return np.zeros_like(np.asarray(t, dtype=float))
    else:
        raise ValueError(f"Unsupported jump process '{process_type}'. Choose 'hawkes' or 'poisson'.")

def Expectation_Nt(t, alpha=27.0, lambda_0=27.0, xi=15.0, beta=9.0, process_type='hawkes'):

    process_type = process_type.lower()

    if process_type == 'hawkes':
        if xi <= beta:
            raise ValueError("Subcriticality condition violated: requires xi > beta.")

        xi_mb = xi - beta
        return (alpha * xi * t) / xi_mb - (1.0 / xi_mb) * (lambda_0 - (alpha * xi) / xi_mb) * (np.exp(-xi_mb * t) - 1.0)

    elif process_type == 'poisson':
        return lambda_0 * t

    else:
        raise ValueError(f"Unsupported jump process '{process_type}'. Choose 'hawkes' or 'poisson'.")


def Var_Nt(t, alpha=27.0, lambda_0=27.0, xi=15.0, beta=9.0, process_type='hawkes'):

    process_type = process_type.lower()

    if process_type == 'hawkes':
        if xi <= beta:
            raise ValueError("Subcriticality condition violated: requires xi > beta.")

        xi_mb = xi - beta
        xi_pb = xi + beta

        xi_mb_2 = xi_mb * xi_mb
        xi_mb_3 = xi_mb_2 * xi_mb
        xi_mb_4 = xi_mb_3 * xi_mb

        c1 = (xi / xi_mb_4) * (
            lambda_0 * xi * xi_mb - alpha * (xi_pb ** 2) + 1.5 * alpha * (beta ** 2)
        )
        c2 = (alpha * (xi ** 3)) / xi_mb_3
        c3 = (1.0 / xi_mb_4) * (
            alpha * xi * (xi_pb ** 2)
            - 2.0 * alpha * (beta ** 2) * xi
            - lambda_0 * xi_mb_2 * xi_pb
        )
        c4 = ((2.0 * xi * beta) / xi_mb_3) * (alpha * xi - lambda_0 * xi_mb)
        c5 = ((beta ** 2) / (2.0 * xi_mb_4)) * (alpha * xi - 2.0 * lambda_0 * xi_mb)

        exp_term = np.exp(-xi_mb * t)
        return c1 + c2 * t + c3 * exp_term + c4 * t * exp_term + c5 * (exp_term ** 2)

    elif process_type == 'poisson':
        return lambda_0 * t

    else:
        raise ValueError(f"Unsupported jump process '{process_type}'. Choose 'hawkes' or 'poisson'.")


# =============================================================================
# Hawkes Process Simulation: Ogata's Thinning Algorithm
# =============================================================================

def lambda_t_computation(t, alpha=27, lambda_0=27, xi=15, beta=9, events=None):

    if events is None or len(events) == 0:
        sums = 0.0
    else:
        events_arr = np.array(events)
        past_events = events_arr[events_arr < t]
        sums = np.sum(np.exp(-xi * (t - past_events))) if len(past_events) > 0 else 0.0

    return alpha + (lambda_0 - alpha) * np.exp(-xi * t) + beta * sums


def Hawkes_Process(T_final=1, alpha=27, lambda_0=27, xi=15, beta=9):  # Ogata's Thinning Algorithm

    events = []
    t = 0.0
    R = 0.0

    while t < T_final:
        mu_t = alpha + max(0.0, lambda_0 - alpha) * np.exp(-xi * t)
        lambda_star = mu_t + beta * R

        U1 = np.random.uniform(0, 1)
        u = -np.log(U1) / lambda_star
        s = t + u

        if s > T_final:
            break

        R_s = R * np.exp(-xi * (s - t))
        lambda_real = alpha + (lambda_0 - alpha) * np.exp(-xi * s) + beta * R_s

        U2 = np.random.uniform(0, 1)
        if U2 <= lambda_real / lambda_star:
            events.append(s)
            R = R_s + 1.0
            t = s
        else:
            R = R_s
            t = s

    return events


# =============================================================================
# Poisson Process Simulation
# =============================================================================

def Poisson_Process(T_final, lambda_0=27):  # Homogeneous

    t = 0.0
    events_time = []
    scale = 1 / lambda_0

    while t <= T_final:
        t += np.random.exponential(scale)
        if t <= T_final:
            events_time.append(t)

    return np.array(events_time)


# =============================================================================
# Unified simulation entry point
# =============================================================================

def Simulate_Process(process_type, T_final=1, alpha=27, lambda_0=27, xi=15, beta=9):
    """Dispatches to Hawkes_Process or Poisson_Process. The two algorithms are
    fundamentally different (thinning vs. direct exponential inter-arrival),
    so they stay as separate functions; this is only a convenience wrapper
    for code that wants a single call point (e.g. the plotting utilities)."""

    process_type = process_type.lower()

    if process_type == 'hawkes':
        return Hawkes_Process(T_final, alpha, lambda_0, xi, beta)
    elif process_type == 'poisson':
        return Poisson_Process(T_final, lambda_0)
    else:
        raise ValueError(f"Unsupported jump process '{process_type}'. Choose 'hawkes' or 'poisson'.")


# =============================================================================
# Visualization
# =============================================================================
