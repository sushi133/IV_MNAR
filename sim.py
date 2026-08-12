"""
Simulation study

Usage:
  python sim.py --n 5000 --reps 1000 --n-jobs 8

Outputs:
  figures/fig_identification_headline.pdf    
  figures/fig_misspecification_heatmap.pdf   
  tables/tab_identification_diagnostics.tex  
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Model setup
# =============================================================================

U_LEVELS: Tuple[str, ...] = ("a", "c", "n")
U_TO_INT = {u: i for i, u in enumerate(U_LEVELS)}
INT_TO_U = {i: u for u, i in U_TO_INT.items()}
VAR_ORDER: Tuple[str, ...] = ("Z", "U", "D", "Y")

IDENTIFIABLE_DGPS: Tuple[str, ...] = ("1ZD", "1UD", "1DY", "1ZY", "1UY")
NONIDENTIFIED_DGPS: Tuple[str, ...] = ("1ZU", "1ZDY", "1UDY", "1ZUY")
ALL_DGPS: Tuple[str, ...] = IDENTIFIABLE_DGPS + NONIDENTIFIED_DGPS

P_Z = 0.5
P_U: Dict[str, float] = {"a": 0.25, "c": 0.50, "n": 0.25}
Q_TRUE: Dict[Tuple[str, int], float] = {
    ("a", 1): 0.05, ("n", 0): 0.65, ("c", 0): 0.10, ("c", 1): 0.65,
}
TRUE_CACE = Q_TRUE[("c", 1)] - Q_TRUE[("c", 0)]
TARGET_RESPONSE = 0.65

DIAG_N = 100_000.0
DIAG_STARTS = 100
DIAG_MAXITER = 2000
TOP_LOGLIK_TOL = 1e-8
PROFILE_MIN, PROFILE_MAX = -1.0, 1.0
PROFILE_GRID = 81
PROFILE_STARTS = 4
PROFILE_MAXITER = 1200

# =============================================================================
# Helpers
# =============================================================================

def mechanism_vars(mech: str) -> Tuple[str, ...]:
    """Variables the mechanism depends on, in canonical order."""
    chars = tuple(mech[1:])
    return tuple(v for v in VAR_ORDER if v in chars)


def d_from_z_u(z: int, u: str) -> int:
    return 1 if u == "a" else 0 if u == "n" else int(z)


def q_y1(u: str, d: int, q: Mapping[Tuple[str, int], float] = Q_TRUE) -> float:
    return q[("a", 1)] if u == "a" else q[("n", 0)] if u == "n" else q[("c", int(d))]


def all_full_states() -> Iterable[Tuple[int, str, int, int, float]]:
    """Yield (z, u, d, y, probability) under the complete-data distribution."""
    for z in (0, 1):
        for u in U_LEVELS:
            d = d_from_z_u(z, u)
            py1 = q_y1(u, d)
            for y in (0, 1):
                py = py1 if y == 1 else 1.0 - py1
                yield z, u, d, y, (P_Z if z == 1 else 1 - P_Z) * P_U[u] * py


def logit_safe(p: float, eps: float = 1e-6) -> float:
    return float(logit(np.clip(p, eps, 1 - eps)))


def default_n_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))

# =============================================================================
# Missingness data-generating mechanisms
# =============================================================================

X_U = {"a": -1.00, "c": 0.55, "n": 1.05}
MAIN_COEF = {"Z": 0.55, "U": -0.80, "D": 0.70, "Y": -1.10}
PAIR_COEF = {
    ("Z", "U"): 0.45, ("Z", "D"): -0.60, ("Z", "Y"): 0.55,
    ("U", "D"): 0.65, ("U", "Y"): -0.75, ("D", "Y"): 0.50,
}
TRIPLE_COEF = {
    ("Z", "U", "Y"): -0.40, ("Z", "D", "Y"): 0.35, ("U", "D", "Y"): -0.35,
}
SCORE_SCALE = 0.45


def x_var(var: str, z: int, u: str, d: int, y: int) -> float:
    if var == "Z":
        return 2.0 * z - 1.0
    if var == "D":
        return 2.0 * d - 1.0
    if var == "Y":
        return 2.0 * y - 1.0
    return X_U[u]


def response_score(mech: str, z: int, u: str, d: int, y: int) -> float:
    """How (Z, U, D, Y) shift the response log-odds, before the calibrated intercept."""
    vars_ = mechanism_vars(mech)
    x = {v: x_var(v, z, u, d, y) for v in vars_}
    score = sum(MAIN_COEF[v] * x[v] for v in vars_)
    for pair in itertools.combinations(vars_, 2):
        score += PAIR_COEF[pair] * x[pair[0]] * x[pair[1]]
    for triple in itertools.combinations(vars_, 3):
        score += TRIPLE_COEF[triple] * x[triple[0]] * x[triple[1]] * x[triple[2]]
    return float(SCORE_SCALE * score)


@lru_cache(maxsize=None)
def calibrate_response_intercept(mech: str, target_response: float) -> float:
    """Calibrate the intercept so that the average response rate matches the target."""
    scores = np.array([response_score(mech, z, u, d, y)
                       for z, u, d, y, _ in all_full_states()])
    weights = np.array([p for *_, p in all_full_states()])
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(np.sum(weights * expit(mid + scores))) < target_response:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=None)
def response_table_for_dgp(mech: str, target_response: float) -> np.ndarray:
    """Table of response probabilities for every (Z, U, D, Y) combination under the mechanism."""
    alpha = calibrate_response_intercept(mech, target_response)
    tab = np.zeros((2, len(U_LEVELS), 2, 2))
    for z in (0, 1):
        for u in U_LEVELS:
            for d in (0, 1):
                for y in (0, 1):
                    tab[z, U_TO_INT[u], d, y] = expit(alpha + response_score(mech, z, u, d, y))
    return tab

# =============================================================================
# Data generation and observed counts
# =============================================================================

def generate_data(n: int, dgp: str, target_response: float,
                  rng: np.random.Generator) -> Dict[str, np.ndarray]:
    z = rng.binomial(1, P_Z, size=n).astype(int)
    u_int = rng.choice(len(U_LEVELS), size=n, p=[P_U[u] for u in U_LEVELS]).astype(int)
    d = np.array([[d_from_z_u(z0, u) for u in U_LEVELS] for z0 in (0, 1)])[z, u_int]
    q_lookup = np.full((len(U_LEVELS), 2), 0.5)
    for u in U_LEVELS:
        for dd in (0, 1):
            if not ((u == "a" and dd == 0) or (u == "n" and dd == 1)):
                q_lookup[U_TO_INT[u], dd] = q_y1(u, dd)
    y = rng.binomial(1, q_lookup[u_int, d]).astype(int)
    r = rng.binomial(1, response_table_for_dgp(dgp, target_response)[z, u_int, d, y]).astype(int)
    return {"Z": z, "U": u_int, "D": d, "Y": y, "R": r}


def aggregate_counts(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Observed-data counts by (Z, D, R, Y)."""
    counts = np.zeros((2, 2, 2, 2))
    yobs = np.where(data["R"] == 1, data["Y"], 0)
    np.add.at(counts, (data["Z"], data["D"], data["R"], yobs), 1.0)
    return counts


def population_counts(mech: str, target_response: float, n_total: float = DIAG_N) -> np.ndarray:
    """Population (expected) observed-data counts under the mechanism, used by the diagnostics."""
    rho = response_table_for_dgp(mech, target_response)
    counts = np.zeros((2, 2, 2, 2))
    for z, u, d, y, prob in all_full_states():
        pr = rho[z, U_TO_INT[u], d, y]
        counts[z, d, 1, y] += n_total * prob * pr
        counts[z, d, 0, 0] += n_total * prob * (1.0 - pr)
    return counts

# =============================================================================
# Estimators
# =============================================================================

def safe_wald_from_joint(p_dy_given_z: np.ndarray) -> float:
    """Wald (IV) estimate of the CACE from the joint law of (D, Y) given Z."""
    ey = np.array([sum(y * p_dy_given_z[d, y, z] for d in (0, 1) for y in (0, 1)) for z in (0, 1)])
    ed = np.array([sum(d * p_dy_given_z[d, y, z] for d in (0, 1) for y in (0, 1)) for z in (0, 1)])
    denom = ed[1] - ed[0]
    return np.nan if abs(denom) < 1e-10 else float((ey[1] - ey[0]) / denom)


def oracle_estimator(data: Mapping[str, np.ndarray]) -> float:
    """Oracle Wald estimate from the complete data, as if nothing were missing."""
    z, d, y = data["Z"], data["D"], data["Y"]
    p = np.zeros((2, 2, 2))
    for zz in (0, 1):
        mask = z == zz
        nz = mask.sum()
        if nz == 0:
            return np.nan
        for dd in (0, 1):
            for yy in (0, 1):
                p[dd, yy, zz] = np.sum(mask & (d == dd) & (y == yy)) / nz
    return safe_wald_from_joint(p)


def observed_components(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed-data pieces given Z: A = P(D, Y, observed), B = P(D, not observed), P_D = P(D)."""
    nz = counts.sum(axis=(1, 2, 3))
    A, B, P_D = np.zeros((2, 2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    for z in (0, 1):
        if nz[z] <= 0:
            A[z], B[z], P_D[z] = np.nan, np.nan, np.nan
            continue
        A[z] = counts[z, :, 1, :] / nz[z]
        B[z] = counts[z, :, 0, 0] / nz[z]
        P_D[z] = counts[z].sum(axis=(1, 2)) / nz[z]
    return A, B, P_D


def estimator_1zd(counts: np.ndarray) -> float:
    """CACE under mechanism 1ZD: reweight the observed cells to undo ignorable outcome missingness."""
    A, _, P_D = observed_components(counts)
    pdy = np.zeros((2, 2, 2))
    for z in (0, 1):
        for d in (0, 1):
            obs = A[z, d, :].sum()
            if obs <= 1e-12 or P_D[z, d] <= 1e-12:
                return np.nan
            pdy[d, :, z] = A[z, d, :] / (obs / P_D[z, d])
    return safe_wald_from_joint(pdy)


def estimator_1dy(counts: np.ndarray) -> float:
    """CACE under mechanism 1DY: solve a 2x2 system within each treatment level to recover the response odds."""
    A, B, _ = observed_components(counts)
    pdy = np.zeros((2, 2, 2))
    for d in (0, 1):
        mat = np.array([[A[0, d, 0], A[0, d, 1]], [A[1, d, 0], A[1, d, 1]]])
        if abs(np.linalg.det(mat)) <= 1e-12:
            return np.nan
        eta = np.linalg.solve(mat, np.array([B[0, d], B[1, d]]))
        for z in (0, 1):
            pdy[d, :, z] = A[z, d, :] * (1.0 + eta)
    return safe_wald_from_joint(pdy)


def estimator_1zy(counts: np.ndarray) -> float:
    """CACE under mechanism 1ZY: solve a 2x2 system within each instrument level to recover the response odds."""
    A, B, _ = observed_components(counts)
    pdy = np.zeros((2, 2, 2))
    for z in (0, 1):
        mat = np.array([[A[z, 0, 0], A[z, 0, 1]], [A[z, 1, 0], A[z, 1, 1]]])
        if abs(np.linalg.det(mat)) <= 1e-12:
            return np.nan
        eta = np.linalg.solve(mat, np.array([B[z, 0], B[z, 1]]))
        for d in (0, 1):
            pdy[d, :, z] = A[z, d, :] * (1.0 + eta)
    return safe_wald_from_joint(pdy)


def estimator_1ud(counts: np.ndarray) -> float:
    """CACE under mechanism 1UD: take differences across the instrument to isolate compliers."""
    A, _, _ = observed_components(counts)
    comp0, comp1 = A[0, 0, :] - A[1, 0, :], A[1, 1, :] - A[0, 1, :]
    den0, den1 = comp0.sum(), comp1.sum()
    if abs(den0) <= 1e-12 or abs(den1) <= 1e-12:
        return np.nan
    return float(comp1[1] / den1 - comp0[1] / den0)


def estimator_1uy(counts: np.ndarray) -> float:
    """CACE under mechanism 1UY: use ratios across the instrument to recover complier outcome means."""
    A, _, _ = observed_components(counts)
    comp0, comp1 = A[0, 0, :] - A[1, 0, :], A[1, 1, :] - A[0, 1, :]
    if abs(comp1[0]) <= 1e-12 or abs(comp1[1]) <= 1e-12:
        return np.nan
    ratio0, ratio1 = comp0[0] / comp1[0], comp0[1] / comp1[1]
    denom = ratio1 - ratio0
    if abs(denom) <= 1e-12:
        return np.nan
    q_c1 = (1.0 - ratio0) / denom
    return float(q_c1 - ratio1 * q_c1)


_ESTIMATORS = {
    "1ZD": estimator_1zd, "1UD": estimator_1ud, "1DY": estimator_1dy,
    "1ZY": estimator_1zy, "1UY": estimator_1uy,
}

# =============================================================================
# Saturated observed-data likelihood (for the no-sampling diagnostics/profiles)
# =============================================================================

def response_cells(mech: str) -> List[Tuple]:
    levels = [tuple(range(len(U_LEVELS))) if v == "U" else (0, 1) for v in mechanism_vars(mech)]
    return list(itertools.product(*levels)) if levels else [tuple()]


def cell_key(mech: str, z: int, u_int: int, d: int, y: int) -> Tuple:
    vals = {"Z": int(z), "U": int(u_int), "D": int(d), "Y": int(y)}
    return tuple(vals[v] for v in mechanism_vars(mech))


def compatible_u(z: int, d: int) -> List[int]:
    return [U_TO_INT[u] for u in U_LEVELS if d_from_z_u(z, u) == d]


def unpack_params(theta: np.ndarray, mech: str):
    pu = softmax(np.array([theta[0], theta[1], 0.0]))
    qr = expit(theta[2:6])
    q = {("a", 1): float(qr[0]), ("c", 0): float(qr[1]),
         ("c", 1): float(qr[2]), ("n", 0): float(qr[3])}
    cells = response_cells(mech)
    r = {cell: float(p) for cell, p in zip(cells, expit(theta[6:6 + len(cells)]))}
    return pu, q, r


def model_probs(theta: np.ndarray, mech: str) -> np.ndarray:
    """Predicted cell probabilities for (Z, D, R, Y) under the saturated model."""
    pu, q, rtab = unpack_params(theta, mech)
    probs = np.zeros((2, 2, 2, 2))
    for z in (0, 1):
        for d in (0, 1):
            for u_int in compatible_u(z, d):
                p_u = pu[u_int]
                py1 = q_y1(INT_TO_U[u_int], d, q)
                for y in (0, 1):
                    py = py1 if y == 1 else 1.0 - py1
                    rho = rtab[cell_key(mech, z, u_int, d, y)]
                    probs[z, d, 1, y] += p_u * py * rho
                    probs[z, d, 0, 0] += p_u * py * (1.0 - rho)
    return probs


def _loglik_from_probs(probs: np.ndarray, counts: np.ndarray) -> float:
    eps = 1e-14
    ll = 0.0
    for z in (0, 1):
        for d in (0, 1):
            for y in (0, 1):
                c = counts[z, d, 1, y]
                if c:
                    ll += c * math.log(max(probs[z, d, 1, y], eps))
            c0 = counts[z, d, 0, 0]
            if c0:
                ll += c0 * math.log(max(probs[z, d, 0, 0], eps))
    return ll


def neg_loglik(theta: np.ndarray, counts: np.ndarray, mech: str) -> float:
    return -_loglik_from_probs(model_probs(theta, mech), counts)


def initial_theta(counts: np.ndarray, mech: str,
                  rng: Optional[np.random.Generator] = None, jitter: float = 0.0) -> np.ndarray:
    nz = counts.sum(axis=(1, 2, 3))
    p_a = np.clip(counts[0, 1].sum() / max(nz[0], 1.0), 0.02, 0.94)
    p_n = np.clip(counts[1, 0].sum() / max(nz[1], 1.0), 0.02, 0.94)
    p_c = np.clip(1.0 - p_a - p_n, 0.02, 0.94)
    s = p_a + p_c + p_n
    p_a, p_c, p_n = p_a / s, p_c / s, p_n / s

    def obs_mean(z: int, d: int, default: float) -> float:
        den = counts[z, d, 1, :].sum()
        return default if den <= 0 else float(np.clip(counts[z, d, 1, 1] / den, 0.05, 0.95))

    q_logits = [logit_safe(obs_mean(0, 1, 0.60)), logit_safe(obs_mean(0, 0, 0.25)),
                logit_safe(obs_mean(1, 1, 0.70)), logit_safe(obs_mean(1, 0, 0.30))]
    overall_r = float(np.clip(counts[:, :, 1, :].sum() / max(counts.sum(), 1.0), 0.05, 0.95))
    theta = np.array([math.log(p_a / p_n), math.log(p_c / p_n)] + q_logits
                     + [logit_safe(overall_r)] * len(response_cells(mech)))
    if rng is not None and jitter > 0:
        theta = theta + rng.normal(scale=jitter, size=theta.size)
    return theta


def fit_mle(counts: np.ndarray, mech: str, n_starts: int, seed, maxiter: int) -> Dict[str, object]:
    """Fit the saturated likelihood from several starting values; return the best fit and all starts."""
    rng = np.random.default_rng(seed)
    starts = [initial_theta(counts, mech)]
    starts += [initial_theta(counts, mech, rng=rng, jitter=1.0) for _ in range(max(0, n_starts - 1))]
    records, best = [], None
    for k, start in enumerate(starts):
        res = minimize(neg_loglik, start, args=(counts, mech), method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 50})
        _, q, _ = unpack_params(res.x, mech)
        rec = {"success": bool(res.success), "fun": float(res.fun),
               "loglik": float(-res.fun), "cace": float(q[("c", 1)] - q[("c", 0)])}
        records.append(rec)
        if best is None or rec["fun"] < best["fun"]:
            best = rec
    return {"cace": best["cace"], "loglik": best["loglik"], "all_starts": records}

# =============================================================================
# Profile likelihood for the CACE 
# =============================================================================

def unpack_profile_params(phi: np.ndarray, mech: str, cace_fixed: float):
    pu = softmax(np.array([phi[0], phi[1], 0.0]))
    low, high = max(0.0, -cace_fixed), min(1.0, 1.0 - cace_fixed)
    width = high - low
    q_c0 = low if width <= 0 else low + width * float(expit(phi[4]))
    q = {("a", 1): float(expit(phi[2])), ("n", 0): float(expit(phi[3])),
         ("c", 0): q_c0, ("c", 1): q_c0 + cace_fixed}
    cells = response_cells(mech)
    r = {cell: float(p) for cell, p in zip(cells, expit(phi[5:5 + len(cells)]))}
    return pu, q, r


def model_probs_profile(phi: np.ndarray, mech: str, cace_fixed: float) -> np.ndarray:
    pu, q, rtab = unpack_profile_params(phi, mech, cace_fixed)
    probs = np.zeros((2, 2, 2, 2))
    for z in (0, 1):
        for d in (0, 1):
            for u_int in compatible_u(z, d):
                p_u = pu[u_int]
                py1 = q_y1(INT_TO_U[u_int], d, q)
                for y in (0, 1):
                    py = py1 if y == 1 else 1.0 - py1
                    rho = rtab[cell_key(mech, z, u_int, d, y)]
                    probs[z, d, 1, y] += p_u * py * rho
                    probs[z, d, 0, 0] += p_u * py * (1.0 - rho)
    return probs


def neg_loglik_profile(phi: np.ndarray, counts: np.ndarray, mech: str, cace_fixed: float) -> float:
    if cace_fixed < -1 or cace_fixed > 1:
        return 1e100
    return -_loglik_from_probs(model_probs_profile(phi, mech, cace_fixed), counts)


def initial_phi(counts: np.ndarray, mech: str, cace_fixed: float,
                rng: Optional[np.random.Generator], jitter: float) -> np.ndarray:
    theta = initial_theta(counts, mech)
    low, high = max(0.0, -cace_fixed), min(1.0, 1.0 - cace_fixed)
    width = high - low
    if width <= 0:
        t_c0 = 0.0
    else:
        frac = np.clip((float(expit(theta[3])) - low) / width, 0.05, 0.95)
        t_c0 = logit_safe(float(frac))
    phi = np.concatenate([theta[0:2], np.array([theta[2], theta[5], t_c0]), theta[6:]])
    if rng is not None and jitter > 0:
        phi = phi + rng.normal(scale=jitter, size=phi.size)
    return phi


def profile_likelihood(counts: np.ndarray, mech: str, grid: Sequence[float],
                       n_starts: int, seed, maxiter: int) -> pd.DataFrame:
    """Profile log-likelihood for the CACE: at each CACE on the grid, maximize over all other parameters."""
    rng = np.random.default_rng(seed)
    rows = []
    for cace_fixed in grid:
        best = np.inf
        for s in range(n_starts):
            phi0 = initial_phi(counts, mech, cace_fixed,
                               rng if s > 0 else None, jitter=1.0 if s > 0 else 0.0)
            res = minimize(neg_loglik_profile, phi0, args=(counts, mech, float(cace_fixed)),
                           method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50})
            best = min(best, float(res.fun))
        rows.append({"cace_grid": float(cace_fixed), "loglik": -best})
    df = pd.DataFrame(rows)
    df["rel_loglik"] = df["loglik"] - df["loglik"].max()
    df["mechanism"] = mech
    return df[["mechanism", "cace_grid", "rel_loglik"]]

# =============================================================================
# Monte Carlo experiment 
# =============================================================================

def _mc_task(task: Tuple[int, str, int, int, int]) -> List[Dict[str, object]]:
    """One Monte Carlo replication for one DGP, evaluating all five estimators."""
    dgp_idx, dgp, rep, n, seed_base = task
    rng = np.random.default_rng(np.random.SeedSequence([int(seed_base), 1, int(dgp_idx), int(rep)]))
    data = generate_data(n, dgp, TARGET_RESPONSE, rng)
    counts = aggregate_counts(data)
    oracle = oracle_estimator(data)
    return [{"dgp": dgp, "estimator": m, "estimate": _ESTIMATORS[m](counts),
             "oracle_estimate": oracle} for m in IDENTIFIABLE_DGPS]


def run_monte_carlo(args: argparse.Namespace) -> pd.DataFrame:
    tasks = [(i, dgp, b, args.n, args.seed)
             for i, dgp in enumerate(ALL_DGPS) for b in range(args.reps)]
    records: List[Dict[str, object]] = []
    t0, total = time.time(), len(tasks)

    def run() -> Iterable[List[Dict[str, object]]]:
        if args.n_jobs <= 1:
            return (_mc_task(t) for t in tasks)
        ex = ProcessPoolExecutor(max_workers=args.n_jobs)
        return ex.map(_mc_task, tasks, chunksize=10)

    for done, rep_rows in enumerate(run(), start=1):
        records.extend(rep_rows)
    df = pd.DataFrame(records)
    df["error_vs_oracle"] = df["estimate"] - df["oracle_estimate"]
    return df


def summarize_monte_carlo(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean excess bias, estimate minus complete-data estimate, by (DGP, fitted mechanism)."""
    rows = [{"dgp": dgp, "estimator": est,
             "excess_bias_vs_oracle": float(np.nanmean(g["error_vs_oracle"]))}
            for (dgp, est), g in raw.groupby(["dgp", "estimator"], sort=False)]
    return pd.DataFrame(rows)

# =============================================================================
# Population likelihood diagnostics and profiles 
# =============================================================================

def _diag_task(task: Tuple[int, str, int]) -> Dict[str, object]:
    i, mech, seed_base = task
    counts = population_counts(mech, TARGET_RESPONSE)
    fit = fit_mle(counts, mech, n_starts=DIAG_STARTS,
                  seed=np.random.SeedSequence([int(seed_base), 2, int(i)]),
                  maxiter=DIAG_MAXITER)
    starts = fit["all_starts"]
    max_ll = max(r["loglik"] for r in starts)
    top = [r["cace"] for r in starts if (max_ll - r["loglik"]) <= TOP_LOGLIK_TOL * DIAG_N]
    return {"mechanism": mech, "identified": mech in IDENTIFIABLE_DGPS,
            "cace_lo": float(min(top)), "cace_hi": float(max(top)),
            "n_starts": len(starts), "n_at_max": len(top)}


def run_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    tasks = [(i, mech, args.seed) for i, mech in enumerate(ALL_DGPS)]
    if args.n_jobs <= 1:
        rows = [_diag_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            rows = list(ex.map(_diag_task, tasks, chunksize=1))
    return pd.DataFrame(rows)


def _profile_task(task: Tuple[int, str, int]) -> pd.DataFrame:
    i, mech, seed_base = task
    grid = np.linspace(PROFILE_MIN, PROFILE_MAX, PROFILE_GRID)
    return profile_likelihood(population_counts(mech, TARGET_RESPONSE), mech, grid,
                              n_starts=PROFILE_STARTS,
                              seed=np.random.SeedSequence([int(seed_base), 3, int(i)]),
                              maxiter=PROFILE_MAXITER)


def run_profiles(args: argparse.Namespace) -> pd.DataFrame:
    tasks = [(i, mech, args.seed) for i, mech in enumerate(ALL_DGPS)]
    if args.n_jobs <= 1:
        frames = [_profile_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            frames = list(ex.map(_profile_task, tasks, chunksize=1))
    return pd.concat(frames, ignore_index=True)

# =============================================================================
# Presentation: figures and LaTeX tables
# =============================================================================
_PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
            "vermillion": "#D55E00", "purple": "#CC79A7", "skyblue": "#56B4E9",
            "yellow": "#F0E442", "grey": "#999999", "black": "#000000"}
_IDENT_COLORS = {"1ZD": _PALETTE["blue"], "1UD": _PALETTE["green"], "1DY": _PALETTE["skyblue"],
                 "1ZY": _PALETTE["purple"], "1UY": _PALETTE["black"]}
_NONIDENT_COLORS = {"1ZU": _PALETTE["vermillion"], "1ZDY": _PALETTE["orange"],
                    "1UDY": _PALETTE["yellow"], "1ZUY": _PALETTE["grey"]}
_IDENT_LINESTYLES = {"1ZD": (0, (1, 1.3)), "1UD": (0, (4.5, 1.8)),
                     "1DY": (0, (5, 1.6, 1, 1.6)), "1ZY": (0, (3, 1.4, 1, 1.4, 1, 1.4)),
                     "1UY": (0, (1, 1))}
_IDENT_LINEWIDTHS = {"1ZD": 2.9, "1UD": 2.3, "1DY": 1.9, "1ZY": 1.5, "1UY": 1.1}


def _mech_label(mech: str) -> str:
    return rf"$1\mathit{{{mech[1:]}}}$"


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 320,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
        "legend.fontsize": 9.0, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.8,
    })


def make_headline_figure(raw: pd.DataFrame, profiles: pd.DataFrame, outpath: str) -> None:
    """Headline figure: (a) sampling distributions under identifiable mechanisms;
    (b) profile log-likelihoods from the true observed-data distribution rather than
    a sample (sharp peak vs. flat ridge)."""
    apply_paper_style()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.0, 5.5),
                                     gridspec_kw={"height_ratios": [1.0, 1.15]})

    # ----- correctly specified sampling distributions -----
    ypos = np.arange(len(IDENTIFIABLE_DGPS))[::-1]
    for m, y0 in zip(IDENTIFIABLE_DGPS, ypos):
        vals = raw[(raw["dgp"] == m) & (raw["estimator"] == m)]["estimate"].dropna().to_numpy()
        if vals.size == 0:
            continue
        color = _IDENT_COLORS[m]
        q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        ax_a.plot([q05, q95], [y0, y0], color=color, lw=1.5, zorder=2)
        for xq in (q05, q95):
            ax_a.plot([xq, xq], [y0 - 0.11, y0 + 0.11], color=color, lw=1.5, zorder=2)
        ax_a.add_patch(plt.Rectangle((q25, y0 - 0.22), q75 - q25, 0.44,
                                     facecolor=color, edgecolor=color, alpha=0.32, zorder=3))
        ax_a.plot([q50, q50], [y0 - 0.22, y0 + 0.22], color=color, lw=2.2, zorder=4)
        ax_a.plot([float(np.mean(vals))], [y0], marker="o", ms=4.8, color=color,
                  markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    ax_a.axvline(TRUE_CACE, color=_PALETTE["black"], ls="--", lw=1.2, zorder=1)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels([_mech_label(m) for m in IDENTIFIABLE_DGPS])
    ax_a.set_ylim(-0.7, len(IDENTIFIABLE_DGPS) - 0.3)
    ax_a.set_xlabel(r"Estimated CACE across Monte Carlo replications")
    ax_a.set_title(r"(a) Estimated CACEs center at the true CACE")
    ax_a.grid(axis="y", visible=False)
    ax_a.set_xlim(0.36, 0.74)
    ax_a.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax_a.margins(x=0.0)

    # ----- profile log-likelihoods from the true observed-data distribution
    present = set(profiles["mechanism"])
    for m in NONIDENTIFIED_DGPS:
        if m in present:
            g = profiles[profiles["mechanism"] == m].sort_values("cace_grid")
            ax_b.plot(g["cace_grid"], g["rel_loglik"], color=_NONIDENT_COLORS[m], lw=1.6,
                      ls="-", solid_capstyle="round", label=_mech_label(m), zorder=3)
    for m in IDENTIFIABLE_DGPS:
        if m in present:
            g = profiles[profiles["mechanism"] == m].sort_values("cace_grid")
            ax_b.plot(g["cace_grid"], g["rel_loglik"], color=_IDENT_COLORS[m],
                      lw=_IDENT_LINEWIDTHS[m], ls=_IDENT_LINESTYLES[m],
                      dash_capstyle="round", label=_mech_label(m), zorder=5)
    ax_b.axvline(TRUE_CACE, color=_PALETTE["black"], ls="--", lw=1.2, zorder=1)
    ax_b.set_xlabel(r"CACE value fixed in the profile")
    ax_b.set_ylabel(r"Profile log-likelihood $-$ maximum")
    ax_b.set_title(r"(b) Identifiable: sharp peak; nonidentifiable: flat ridge")
    ax_b.set_ylim(-16.0, 1.6)
    ax_b.set_xlim(-0.40, 1.0)
    ax_b.set_xticks(np.round(np.arange(-0.3, 1.001, 0.1), 1))
    ax_b.set_yticks([0, -2, -4, -6, -8, -10, -12])
    ax_b.margins(x=0.0)
    handles, labels = ax_b.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [_mech_label(m) for m in list(IDENTIFIABLE_DGPS) + list(NONIDENTIFIED_DGPS)
             if _mech_label(m) in by_label]
    leg = ax_b.legend([by_label[l] for l in order], order, ncol=3, fontsize=9.0,
                      loc="lower left", bbox_to_anchor=(0.015, 0.03),
                      columnspacing=1.3, handlelength=2.6, borderaxespad=0.3,
                      frameon=True, facecolor="white", edgecolor="none", framealpha=0.82)
    leg.set_zorder(6)
    leg._legend_box.align = "left"

    fig.tight_layout(h_pad=1.8)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def make_misspecification_figure(summary: pd.DataFrame, outpath: str) -> None:
    """Heatmap of excess bias: true mechanism (rows) versus fitted model (columns)."""
    apply_paper_style()
    pivot = summary.pivot(index="dgp", columns="estimator", values="excess_bias_vs_oracle")
    rows = [m for m in ALL_DGPS if m in pivot.index]
    cols = [m for m in IDENTIFIABLE_DGPS if m in pivot.columns]
    data = pivot.loc[rows, cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.9))
    finite = np.abs(data[np.isfinite(data)])
    vmax = max(float(np.nanpercentile(finite, 90)) if finite.size else 1.0, 0.30)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([_mech_label(c) for c in cols])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([_mech_label(r) for r in rows])
    ax.set_xlabel("Fitted (identifiable) model")
    ax.set_ylabel("True mechanism")
    ax.grid(visible=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = _fmt(val)
            shade = 0.0 if not np.isfinite(val) else abs(val) / vmax
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shade > 0.55 else "black")
            if rows[i] == cols[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=_PALETTE["black"], lw=1.8))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Excess bias")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def _fmt_range(lo: float, hi: float, nd: int = 2) -> str:
    """Range of CACE estimates over the random starts."""
    if any(v is None or not math.isfinite(float(v)) for v in (lo, hi)):
        return "--"
    return rf"$({_fmt(lo, nd)},\, {_fmt(hi, nd)})$"


def write_diagnostics_table(diag: pd.DataFrame, outpath: str) -> None:
    """Write the diagnostics table: identifiability and the width of the interval of CACE values."""
    by = {r["mechanism"]: r for _, r in diag.iterrows()}
    lines = [r"\begin{tabular}{l c c}", r"\toprule",
             r"Mechanism & Identifiable & Range of CACE estimates \\", r"\midrule"]
    wrote_rule = False
    for m in list(IDENTIFIABLE_DGPS) + list(NONIDENTIFIED_DGPS):
        if m not in by:
            continue
        if m in NONIDENTIFIED_DGPS and not wrote_rule:
            lines.append(r"\midrule")
            wrote_rule = True
        r = by[m]
        ident = "Yes" if bool(r["identified"]) else "No"
        lines.append(f"{_mech_label(m)} & {ident} & "
                     f"{_fmt_range(r['cace_lo'], r['cace_hi'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IV/CACE outcome-MNAR simulation experiment")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--outdir", type=str, default="sim_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.outdir, "figures")
    tab_dir = os.path.join(args.outdir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    t0 = time.time()
    raw = run_monte_carlo(args)
    summary = summarize_monte_carlo(raw)
    diag = run_diagnostics(args)
    profiles = run_profiles(args)

    make_headline_figure(raw, profiles, os.path.join(fig_dir, "fig_identification_headline.pdf"))
    make_misspecification_figure(summary, os.path.join(fig_dir, "fig_misspecification_heatmap.pdf"))
    write_diagnostics_table(diag, os.path.join(tab_dir, "tab_identification_diagnostics.tex"))

if __name__ == "__main__":
    main()
