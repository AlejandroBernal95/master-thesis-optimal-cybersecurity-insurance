# Cyber-Insurance and Security Investment under Hawkes Attack Processes

**Francisco Alejandro Bernal Cañez** — Master's Thesis in Data Science,
Università degli Studi di Padova, 2026.

Code accompanying my Master's thesis in Data Science, on optimal cybersecurity
investment and cyber-insurance design when cyberattacks arrive according to a
**Hawkes self-exciting process** rather than a standard Poisson process. The
thesis extends a risk-neutral HJB-PIDE framework from the literature by
adding coinsurance, a deductible, Lomax/Lognormal loss severities, and a
CARA risk-averse extension, and compares the resulting optimal policies
against the classical Poisson benchmark throughout.

The repository contains the full numerical pipeline used to produce every
figure and table in the thesis: the two HJB-PIDE solvers (risk-neutral and
risk-averse), the premium/elasticity calculations, and the plotting code.

## Repository structure

| File | What it contains |
|---|---|
| `processes.py` | Hawkes and Poisson point processes: closed-form moments (`E[N_t]`, `Var(N_t)`, `E[λ_t]`) and Ogata's thinning simulation. |
| `premiums.py` | Gordon-Loeb security breach probability, and premium/elasticity calculations (Standard Deviation Principle / Variance Principle) for Lomax and Lognormal loss severities. |
| `pide_common.py` | Shared numerical utilities (Jacobian sparsity pattern, solver progress logging) used by both PIDE solvers. |
| `pide_risk_neutral.py` | Risk-neutral HJB-PIDE, under Hawkes and Poisson dynamics, plus the optimal coinsurance/investment extraction. |
| `pide_risk_averse.py` | CARA risk-averse HJB-PIDE (certainty equivalent $W$), including the auxiliary $Q$, $M$ bounds used for numerical stability. |
| `validation.py` | Independent verification of the risk-neutral optimal control, by two methods that never touch the PIDE machinery: a static-control benchmark (direct numerical re-optimization) and closed-loop ODE evaluation under Poisson dynamics. |
| `plotting.py` | All figures used in the thesis: value functions, optimal control surfaces, insurance decision maps, premium comparisons, Hawkes-vs-Poisson gain plots. |
| `discarded_experiments.py` | Functional but unused code: an earlier Pareto Type I loss severity (superseded by Lomax/Lognormal) and an earlier Poisson-only decision framework, kept for reference rather than deleted. |
| `Thesis_Experiments.ipynb` | The notebook that runs every experiment and generates every figure in the thesis, using the modules above. |

## Running the code

Requirements: `numpy`, `scipy`, `matplotlib`. No GPU or special hardware
needed — the PIDE solves use `scipy.integrate.solve_ivp` (Radau method) with
a sparse Jacobian, and run in a few minutes each on a standard laptop.

```bash
pip install numpy scipy matplotlib
jupyter notebook Thesis_Experiments.ipynb
```

The notebook caches solved PIDE tensors to disk (pickle) so repeated runs
don't re-solve from scratch, and saves all generated figures to
`thesis_plots/`.

## Notes

- `validation.py` and `discarded_experiments.py` are not part of the core
  pipeline — they exist for transparency (showing how the optimal control
  was cross-checked, and what earlier modeling choices were tried and set
  aside) rather than to be run as part of a normal notebook pass.
- Notation and modeling choices follow the thesis text; the PDF is the
  authoritative reference for the mathematical derivations behind this code.

## Citation

If you use this code, please cite the thesis:

> Bernal Cañez, Francisco Alejandro. *Cyber-Insurance and Security Investment
> under Hawkes Attack Processes*. Master's Thesis, Università degli Studi di
> Padova, 2026.

```bibtex
@mastersthesis{bernalcanez2026cyberinsurance,
  author = {Bernal Cañez, Francisco Alejandro},
  title  = {Cyber-Insurance and Security Investment under Hawkes Attack Processes},
  school = {Università degli Studi di Padova},
  year   = {2026}
}
```
