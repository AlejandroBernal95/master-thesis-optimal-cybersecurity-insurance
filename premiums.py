"""
Gordon-Loeb security breach probability functions, aggregate-loss premium
calculation (Standard Deviation Principle / Variance Principle), and
elasticity analysis, for losses distributed as Lomax or Lognormal random
variables. (Pareto Type I support moved to discarded_experiments.py --
never part of the thesis' final contributions, which use Lomax and
Lognormal only.)

Aggregate loss over [0,t] is a compound process S_t = sum_{i=1}^{N_t} X_i,
with Var(S_t) = E[N_t]*Var(X_retained) + Var(N_t)*E[X_retained]^2 (standard
compound-process variance decomposition). X_retained is the loss net of
coinsurance C and deductible D: (1-C)*(X-D) + D for X>D, X for X<=D.
"""

import numpy as np
from scipy.stats import norm

from processes import Expectation_Nt, Var_Nt


# =============================================================================
# Security Breach Probability Functions (Gordon-Loeb)
# =============================================================================

def GLI(z, v=0.65, a=0.1, b=1):
    return v / ((a * z + 1) ** b)


def GLII(z, v=0.65, a=0.1):  # In case we want to try another SBPF class in the future.
    return v ** (a * z + 1)  # Dropped for now: the visual results were basically the same.


# =============================================================================
# Shared plumbing (validation + SDP/VP application) -- identical across the
# three loss distributions in the original notebook, factored out once.
# =============================================================================

def _validate_C_and_deductible(C, D, d):
    if not (0 <= C <= 1):
        raise ValueError("C has to be in the interval [0,1].")
    if (D is None) == (d is None):
        raise ValueError("Exactly one of 'D' or 'd' must be specified (the other must be None).")


def _resolve_lambda(lambda_0, lambda_poisson, process_type):
    """lambda_0 is the Hawkes initial intensity; under process_type='poisson'
    it has been silently reused as the constant Poisson rate as well, even
    though the two are conceptually distinct. lambda_poisson lets a Poisson
    call site supply its own rate instead. Backward compatible: if
    lambda_poisson is not given, lambda_0 is used for both process types,
    exactly as before."""
    if lambda_poisson is not None and str(process_type).lower() == 'poisson':
        return lambda_poisson
    return lambda_0


def _apply_premium_principle(A, exp_Nt, loading, C, theta, Principle):
    if Principle == 'SDP':
        if loading < 0:
            raise ValueError("Mathematical error: The expression inside the square root is negative.")
        sqrt_val = np.sqrt(loading)
        Pi = C * (A * exp_Nt + theta * sqrt_val)
        return Pi, sqrt_val

    elif Principle == 'VP':
        Pi = C * A * exp_Nt + theta * loading * (C ** 2)
        return Pi, loading

    else:
        raise ValueError("Premium computation has to follow either 'SDP' or 'VP'.")


# =============================================================================
# Losses distributed as a Lomax (Pareto Type II) Random Variable
# =============================================================================

def _compute_premium_lomax_intermediates(t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta,
                                          D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    if k <= 2:
        raise ValueError("The k parameter has to be larger than 2")
    _validate_C_and_deductible(C, D, d)
    lambda_0 = _resolve_lambda(lambda_0, lambda_poisson, process_type)

    if d is not None:
        if not (0 <= d <= 1):
            raise ValueError("d has to be in the interval [0, 1].")
        D = d * eta_0 / (k - 1)
    else:
        if D < 0:
            raise ValueError("D (Deductible) must be non-negative.")

    exp_Nt = Expectation_Nt(t, alpha, lambda_0, xi, beta, process_type=process_type)
    var_Nt = Var_Nt(t, alpha, lambda_0, xi, beta, process_type=process_type)
    S = GLI(H_0, v, a, b)
    A = (S * (eta_0 ** k)) / ((k - 1.0) * ((eta_0 + D) ** (k - 1.0)))

    loading = ((2.0 * A * (eta_0 + D)) / (k - 2.0) - A ** 2) * exp_Nt + var_Nt * (A ** 2)

    Pi, sqrt_or_loading = _apply_premium_principle(A, exp_Nt, loading, C, theta, Principle)
    return Pi, A, exp_Nt, var_Nt, S, sqrt_or_loading


def calculate_premium_lomax(t, k, eta_0, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                             D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    Pi, _, _, _, _, _ = _compute_premium_lomax_intermediates(
        t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        lambda_poisson=lambda_poisson,
    )
    return Pi


def elasticity_premium_eta_0_lomax(t, k, eta_0, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                   D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    Pi, A, exp_Nt, var_Nt, _, sqrt_or_loading = _compute_premium_lomax_intermediates(
        t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        lambda_poisson=lambda_poisson,
    )

    if Pi == 0:
        return 0.0

    if Principle == 'SDP':
        S = sqrt_or_loading

        if d is not None:
            term_exp = ((2.0 * theta * eta_0 * (k - 1.0 + d)) / (S * (k - 1.0) * (k - 2.0)) - (theta * A / S) + 1.0) * exp_Nt
            term_var = (theta * A / S) * var_Nt
            epsilon_eta_0 = (C * A / Pi) * (term_exp + term_var)
            # From this particular case we expect to have elasticity 1.0 for all pair of parameters (k,eta_0)
            # (See that the relation of eta_0 with Pi is linear)
        else:
            term_exp = (
                (theta * (2.0 * eta_0 + k * D)) / (S * (k - 2.0))
                - (theta * A * (eta_0 + k * D)) / (S * (eta_0 + D))
                + (eta_0 + k * D) / (eta_0 + D)
            ) * exp_Nt
            term_var = (theta * A * (eta_0 + k * D)) / (S * (eta_0 + D)) * var_Nt
            epsilon_eta_0 = (C * A / Pi) * (term_exp + term_var)

        return float(epsilon_eta_0)

    elif Principle == 'VP':
        if d is not None:
            term_exp = ((4.0 * theta * C * eta_0 * (k - 1.0 + d)) / ((k - 1.0) * (k - 2.0)) - 2.0 * theta * C * A + 1.0) * exp_Nt
            term_var = 2.0 * theta * C * A * var_Nt
            epsilon_eta_0 = (C * A / Pi) * (term_exp + term_var)
        else:
            term_exp = (
                (2.0 * theta * C * (2.0 * eta_0 + k * D)) / (k - 2.0)
                - (2.0 * theta * C * A * (eta_0 + k * D)) / (eta_0 + D)
                + (eta_0 + k * D) / (eta_0 + D)
            ) * exp_Nt
            term_var = (2.0 * theta * C * A * (eta_0 + k * D)) / (eta_0 + D) * var_Nt
            epsilon_eta_0 = (C * A / Pi) * (term_exp + term_var)

        return float(epsilon_eta_0)

    else:
        raise ValueError("Principle must be either 'SDP' or 'VP'.")


def elasticity_premium_k_lomax(t, k, eta_0, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                               D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    Pi, A, exp_Nt, var_Nt, _, sqrt_or_loading = _compute_premium_lomax_intermediates(
        t, k, eta_0, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        lambda_poisson=lambda_poisson,
    )

    if Pi == 0:
        return 0.0

    if d is not None:
        log_term = np.log((k - 1.0) / (k - 1.0 + d)) + (k * d - k - 2.0 * d + 1.0) / ((k - 1.0) * (k - 1.0 + d))
    else:
        log_term = np.log(eta_0 / (eta_0 + D)) - 1.0 / (k - 1.0)

    if Principle == 'SDP':
        S = sqrt_or_loading

        if d is not None:
            term_exp = (
                log_term * ((theta * eta_0 * (k - 1.0 + d)) / (S * (k - 1.0) * (k - 2.0)) - (theta * A) / S + 1.0)
                - (theta * eta_0 * (k**2 - 2.0 * k + 1.0 + 2.0 * k * d - 3.0 * d)) / (S * ((k - 1.0)**2) * ((k - 2.0)**2))
            ) * exp_Nt
            term_var = log_term * (theta * A / S) * var_Nt
        else:
            term_exp = (
                log_term * ((theta * (eta_0 + D)) / (S * (k - 2.0)) - (theta * A) / S + 1.0)
                - (theta * (eta_0 + D)) / (S * ((k - 2.0)**2))
            ) * exp_Nt
            term_var = log_term * (theta * A / S) * var_Nt

        epsilon_k = (C * A * k / Pi) * (term_exp + term_var)
        return float(epsilon_k)

    elif Principle == 'VP':
        if d is not None:
            term_exp = (
                log_term * ((2.0 * theta * C * eta_0 * (k - 1.0 + d)) / ((k - 1.0) * (k - 2.0)) - 2.0 * theta * C * A + 1.0)
                - (2.0 * theta * C * eta_0 * (k**2 - 2.0 * k + 1.0 + 2.0 * k * d - 3.0 * d)) / (((k - 1.0)**2) * ((k - 2.0)**2))
            ) * exp_Nt
            term_var = 2.0 * theta * C * A * log_term * var_Nt
        else:
            term_exp = (
                log_term * ((2.0 * theta * C * (eta_0 + D)) / (k - 2.0) - 2.0 * theta * C * A + 1.0)
                - (2.0 * theta * C * (eta_0 + D)) / ((k - 2.0)**2)
            ) * exp_Nt
            term_var = 2.0 * theta * C * A * log_term * var_Nt

        epsilon_k = (C * A * k / Pi) * (term_exp + term_var)
        return float(epsilon_k)

    else:
        raise ValueError("Principle must be either 'SDP' or 'VP'.")


# =============================================================================
# Losses distributed as a Lognormal Random Variable
# =============================================================================

def _compute_premium_lognormal_intermediates(t, mu, sigma, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                             D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    if sigma <= 0:
        raise ValueError("Sigma (volatility) must be strictly positive")
    _validate_C_and_deductible(C, D, d)
    lambda_0 = _resolve_lambda(lambda_0, lambda_poisson, process_type)

    if d is not None:
        if not (0 <= d <= 1):
            raise ValueError("d has to be in the interval [0, 1].")
        D = d * np.exp(mu + (sigma**2) / 2)
    else:
        if D < 0:
            raise ValueError("D (Deductible) must be non-negative.")

    d_LN = (mu - np.log(D)) / sigma
    exp_Nt = Expectation_Nt(t, alpha, lambda_0, xi, beta, process_type=process_type)
    var_Nt = Var_Nt(t, alpha, lambda_0, xi, beta, process_type=process_type)
    S = GLI(H_0, v, a, b)

    M1 = np.exp(mu + (sigma**2) / 2) * norm.cdf(d_LN + sigma) - D * norm.cdf(d_LN)
    A = S * M1

    term_2a = np.exp(2 * mu + 2 * (sigma**2)) * norm.cdf(d_LN + 2 * sigma)
    term_2b = 2 * D * np.exp(mu + (sigma**2) / 2) * norm.cdf(d_LN + sigma)
    term_2c = (D**2) * norm.cdf(d_LN)
    M2 = term_2a - term_2b + term_2c

    loading = (S * M2 - A**2) * exp_Nt + var_Nt * A**2

    Pi, sqrt_or_loading = _apply_premium_principle(A, exp_Nt, loading, C, theta, Principle)
    return Pi, A, M1, M2, d_LN, exp_Nt, var_Nt, S, sqrt_or_loading


def calculate_premium_lognormal(t, mu, sigma, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                                D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    Pi, _, _, _, _, _, _, _, _ = _compute_premium_lognormal_intermediates(
        t, mu, sigma, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        lambda_poisson=lambda_poisson,
    )
    return Pi


def elasticity_premium_mu(t, mu, sigma, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                          D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    Pi, A, M1, M2, d_LN, exp_Nt, var_Nt, S, sqrt_or_loading = _compute_premium_lognormal_intermediates(
        t, mu, sigma, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        lambda_poisson=lambda_poisson,
    )

    if Pi == 0:
        return 0.0

    if d is not None:
        dM1_dmu = M1
        dM2_dmu = 2.0 * M2
    else:
        pdf_d = norm.pdf(d_LN)
        pdf_d_sigma = norm.pdf(d_LN + sigma)
        pdf_d_2sigma = norm.pdf(d_LN + 2.0 * sigma)

        dM1_dmu = np.exp(mu + (sigma**2) / 2.0) * (norm.cdf(d_LN + sigma) + pdf_d_sigma / sigma) - (D * pdf_d) / sigma

        term_2a = np.exp(2.0 * mu + 2.0 * (sigma**2)) * (2.0 * norm.cdf(d_LN + 2.0 * sigma) + pdf_d_2sigma / sigma)
        term_2b = 2.0 * D * np.exp(mu + (sigma**2) / 2.0) * (norm.cdf(d_LN + sigma) + pdf_d_sigma / sigma)
        term_2c = (D**2) * pdf_d / sigma
        dM2_dmu = term_2a - term_2b + term_2c

    if Principle == 'SDP':
        sqrt_val = sqrt_or_loading

        if d is not None:
            term_exp = (theta * S * M2 / sqrt_val - theta * (A**2) / sqrt_val + A) * exp_Nt
            term_var = (theta * (A**2) / sqrt_val) * var_Nt
            epsilon_mu = (C * mu / Pi) * (term_exp + term_var)
        else:
            term_exp = ((theta / sqrt_val) * dM2_dmu - (theta * A / sqrt_val) * dM1_dmu + dM1_dmu) * exp_Nt
            term_var = (theta * A / sqrt_val) * dM1_dmu * var_Nt
            epsilon_mu = (C * mu * S / Pi) * (term_exp + term_var)

        return float(epsilon_mu)

    elif Principle == 'VP':
        if d is not None:
            term_exp = (2.0 * theta * C * S * M2 - 2.0 * theta * C * (A**2) + A) * exp_Nt
            term_var = 2.0 * theta * C * (A**2) * var_Nt
            epsilon_mu = (C * mu / Pi) * (term_exp + term_var)
        else:
            term_exp = (theta * C * dM2_dmu - 2.0 * theta * C * A * dM1_dmu + dM1_dmu) * exp_Nt
            term_var = 2.0 * theta * C * A * dM1_dmu * var_Nt
            epsilon_mu = (C * mu * S / Pi) * (term_exp + term_var)

        return float(epsilon_mu)

    else:
        raise ValueError("Principle must be either 'SDP' or 'VP'.")


def elasticity_premium_sigma(t, mu, sigma, C, theta=0.3, H_0=0, v=0.65, a=0.1, b=1, alpha=27, lambda_0=27, xi=15, beta=9,
                            D=None, d=None, Principle=None, process_type=None, lambda_poisson=None):

    _validate_C_and_deductible(C, D, d)

    Pi, A, M1, M2, d_LN, exp_Nt, var_Nt, S, sqrt_or_loading = _compute_premium_lognormal_intermediates(
        t, mu, sigma, C, theta, H_0, v, a, b, alpha, lambda_0, xi, beta, D, d, Principle, process_type,
        lambda_poisson=lambda_poisson,
    )

    if Pi == 0:
        return 0.0

    pdf_d = norm.pdf(d_LN)
    pdf_d_sigma = norm.pdf(d_LN + sigma)
    pdf_d_2sigma = norm.pdf(d_LN + 2.0 * sigma)

    if d is not None:
        term_d1_a = (np.log(d) / (sigma**2) + 0.5) * pdf_d_sigma
        term_d1_b = d * (np.log(d) / (sigma**2) - 0.5) * pdf_d
        dM1_dsigma = sigma * M1 + np.exp(mu + (sigma**2) / 2.0) * (term_d1_a - term_d1_b)

        term_d2_a = 2.0 * sigma * np.exp(sigma**2) * norm.cdf(d_LN + 2.0 * sigma)
        term_d2_b = np.exp(sigma**2) * pdf_d_2sigma * (np.log(d) / (sigma**2) + 1.5)
        term_d2_c = 2.0 * d * pdf_d_sigma * (np.log(d) / (sigma**2) + 0.5)
        term_d2_d = (d**2) * pdf_d * (np.log(d) / (sigma**2) - 0.5)
        dM2_dsigma = 2 * sigma * M2 + np.exp(2.0 * (mu + (sigma**2) / 2.0)) * (term_d2_a + term_d2_b - term_d2_c + term_d2_d)

    else:
        d_LN_prime = -(mu - np.log(D)) / (sigma**2)

        dM1_dsigma = np.exp(mu + (sigma**2) / 2.0) * (
            sigma * norm.cdf(d_LN + sigma) + (1.0 + d_LN_prime) * pdf_d_sigma
        ) - D * pdf_d * d_LN_prime

        term_D2_a = np.exp(2.0 * mu + 2.0 * (sigma**2)) * (
            4.0 * sigma * norm.cdf(d_LN + 2.0 * sigma) + (2.0 + d_LN_prime) * pdf_d_2sigma
        )
        term_D2_b = 2.0 * D * np.exp(mu + (sigma**2) / 2.0) * (
            sigma * norm.cdf(d_LN + sigma) + (1.0 + d_LN_prime) * pdf_d_sigma
        )
        term_D2_c = (D**2) * pdf_d * d_LN_prime
        dM2_dsigma = term_D2_a - term_D2_b + term_D2_c

    factor = (C * sigma * S) / Pi

    if Principle == 'SDP':
        sqrt_val = sqrt_or_loading
        term_exp = ((theta / (2.0 * sqrt_val)) * dM2_dsigma - (theta * A / sqrt_val) * dM1_dsigma + dM1_dsigma) * exp_Nt
        term_var = (theta * A / sqrt_val) * dM1_dsigma * var_Nt
        epsilon_sigma = factor * (term_exp + term_var)
        return float(epsilon_sigma)

    elif Principle == 'VP':
        term_exp = (theta * C * dM2_dsigma - 2.0 * theta * C * A * dM1_dsigma + dM1_dsigma) * exp_Nt
        term_var = 2.0 * theta * C * A * dM1_dsigma * var_Nt
        epsilon_sigma = factor * (term_exp + term_var)
        return float(epsilon_sigma)

    else:
        raise ValueError("Principle must be either 'SDP' or 'VP'.")
