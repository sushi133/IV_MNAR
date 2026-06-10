"""
NJCS application

Usage:
  python code/njcs.py --B 500 --n-jobs 8

Outputs:
  njcs_cace_forest.png
  njcs_manuscript_numbers.tex        
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, math, multiprocessing as mp, re, warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import digamma, expit, gammaln, logsumexp, polygamma

warnings.filterwarnings("ignore", category=RuntimeWarning)


def lsig(x): return -np.logaddexp(0.0, -x)
def l1sig(x): return -np.logaddexp(0.0, x)


def solve_small(A, b):
    A = np.asarray(A, float); b = np.asarray(b, float)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return np.zeros_like(b)
    try:
        return np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A, rcond=1e-8) @ b


# Weighted logistic and Gamma (log-link) regressions used to fit the model pieces in the EM steps.
def fit_logit(X, y, w, b0=None, ridge=1e-7, maxit=12, tol=1e-7):
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    keep = w > 1e-12
    p = X.shape[1]
    if keep.sum() == 0:
        return np.zeros(p) if b0 is None else b0.copy()
    X, y, w = X[keep], y[keep], w[keep]
    b = np.zeros(p) if b0 is None or len(b0) != p else b0.copy()
    if np.unique(y).size < 2:
        pr = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
        b[:] = 0.0; b[0] = math.log(pr / (1 - pr)); return b
    if b0 is None:
        pr = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
        b[0] = math.log(pr / (1 - pr))
    pen = np.ones(p); pen[0] = 0.0
    for _ in range(maxit):
        e = np.clip(X @ b, -35, 35)
        pr = expit(e)
        g = X.T @ (w * (y - pr)) - ridge * pen * b
        H = X.T @ ((w * pr * (1 - pr))[:, None] * X) + ridge * np.diag(pen)
        step = solve_small(H, g)
        old = np.sum(w * (y * lsig(e) + (1 - y) * l1sig(e))) - 0.5 * ridge * np.sum(pen * b * b)
        fac = 1.0
        for _ in range(16):
            cand = b + fac * step
            ee = np.clip(X @ cand, -35, 35)
            val = np.sum(w * (y * lsig(ee) + (1 - y) * l1sig(ee))) - 0.5 * ridge * np.sum(pen * cand * cand)
            if np.all(np.isfinite(cand)) and val >= old - 1e-9:
                b = cand; break
            fac *= 0.5
        if np.max(np.abs(fac * step)) < tol: break
    return b


def fit_gamma(X, y, w, b0=None, nu0=None, ridge=1e-8, maxit=14, tol=1e-7):
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    keep = (w > 1e-12) & np.isfinite(y) & (y > 0)
    p = X.shape[1]
    if keep.sum() == 0:
        return np.zeros(p) if b0 is None else b0.copy(), 1.0
    X, y, w = X[keep], y[keep], w[keep]
    if b0 is None or len(b0) != p:
        sw = np.sqrt(w)
        b = solve_small((X * sw[:, None]).T @ (X * sw[:, None]) + 1e-6 * np.eye(p),
                        (X * sw[:, None]).T @ (np.log(y) * sw))
    else:
        b = b0.copy()
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        ratio = y / np.exp(eta)
        g = X.T @ (w * (1 - ratio)); g[1:] += ridge * b[1:]
        H = X.T @ ((w * ratio)[:, None] * X); H[1:, 1:] += ridge * np.eye(p - 1)
        step = solve_small(H, g)
        old = np.sum(w * (eta + ratio)) + 0.5 * ridge * np.sum(b[1:] ** 2)
        fac = 1.0
        for _ in range(16):
            cand = b - fac * step
            ee = np.clip(X @ cand, -30, 30)
            val = np.sum(w * (ee + y / np.exp(ee))) + 0.5 * ridge * np.sum(cand[1:] ** 2)
            if np.all(np.isfinite(cand)) and val <= old + 1e-9:
                b = cand; break
            fac *= 0.5
        if np.max(np.abs(fac * step)) < tol: break
    mu = np.exp(np.clip(X @ b, -30, 30))
    A = float(np.average(np.log(y / mu) - y / mu, weights=w))
    if nu0 is None:
        cv2 = np.average(((y - mu) / np.maximum(mu, 1e-12)) ** 2, weights=w)
        nu = float(np.clip(1 / max(cv2, 1e-3), 0.05, 200))
    else:
        nu = float(np.exp(nu0))
    for _ in range(40):
        f = math.log(nu) - float(digamma(nu)) + 1 + A
        fp = 1 / nu - float(polygamma(1, nu))
        if abs(f) < 1e-9: break
        cand = nu - f / fp if fp else nu / 2
        nu = float(np.clip(cand if np.isfinite(cand) and cand > 0 else nu / 2, 1e-4, 1e5))
    return b, nu


# Build the analysis dataset: instrument Z, treatment D, weekly earnings Y (both subject to missingness), and covariates X.
def load_njcs(data_dir, use_covariates=True):
    f1 = os.path.join(data_dir, "mpr_jobcorps_team5_nrw_upd_r_nositeid.dta")
    f2 = os.path.join(data_dir, "key_vars.dta")
    f3 = os.path.join(data_dir, "jobcorps_everjc30.dta")
    for f in (f1, f2, f3):
        if not os.path.exists(f): raise FileNotFoundError(f)
    d = pd.read_stata(f1, convert_categoricals=False).merge(
        pd.read_stata(f2, convert_categoricals=False), on="mprid", suffixes=(".x", ".y")
    ).merge(pd.read_stata(f3, convert_categoricals=False), on="mprid", how="left")

    out = pd.DataFrame({"Z": d["treatmnt.x"].astype(int)})
    # One-sided noncompliance: D=0 whenever Z=0; for Z=1, D is enrollment (may be missing).
    out["D"] = np.where(out.Z.to_numpy() == 0, 0.0, d["everjc30"].astype(float).to_numpy())
    out["Y"] = d["aearny4"].astype(float) / 52.0          # weekly earnings = annual earnings / 52
    out["RD"] = (~pd.isna(out.D)).astype(int)              # treatment-observed indicator
    out["RY"] = (~pd.isna(out.Y)).astype(int)              # outcome-observed indicator
    out["H"] = np.where(out.RY.to_numpy() == 1, (out.Y.to_numpy() > 0).astype(int), -1)  # H = 1(Y>0): the positive-earnings part of the two-part model

    arrst_var = "miss_arrst" if "miss_arrst" in d.columns else "miss_arrst_upd"
    arrst = np.where(d[arrst_var].fillna(0).to_numpy() == 1, 2, d["arrst"].astype(float).to_numpy())
    educ = np.where(pd.isna(d["EDUC_GR"]), 4, d["EDUC_GR"])
    earnb = np.full(len(d), np.nan)
    for k in range(5): earnb = np.where(d[f"earnb{k}"].to_numpy() == 1, k, earnb)
    earnb = np.where(pd.isna(earnb), 5, earnb)
    age = np.where(d["age2024"].to_numpy() == 1, 2, d["age1819"].astype(float).to_numpy())
    race = d["race_b"].astype(float).to_numpy()
    race = np.where(d["race_h"].to_numpy() == 1, 2, race)
    race = np.where(d["race_o"].to_numpy() == 1, 3, race)
    cov = pd.DataFrame({"female": d["female.x"].astype(float), "age": age, "race": race,
                        "haschld": d["haschld_upd"].astype(float), "arrst": arrst,
                        "educ": educ, "earnb": earnb})
    # Missing covariate values are coded as an explicit extra category, so missingness in X
    # is absorbed into its own indicator rather than dropping or imputing the unit.
    for c, miss in {"female": 2, "age": 3, "race": 4, "haschld": 2, "arrst": 2, "educ": 4, "earnb": 5}.items():
        cov[c] = np.where(pd.isna(cov[c]), miss, cov[c])
        out[c] = cov[c]
    if not use_covariates:
        return out, np.ones((len(out), 1))
    X = pd.get_dummies(cov.astype("category"), drop_first=True, dtype=float)  # one-hot encode; the missing categories become their own dummies
    X.insert(0, "Intercept", 1.0)
    return out, X.to_numpy(float)


# The 13 nonredundant mechanisms compatible with one-sided noncompliance in the NJCS.
# For each mechanism, rd lists the variables the treatment-response model R^D depends on,
# and ry lists the variables the outcome-response model R^Y depends on (beyond covariates X).
@dataclass(frozen=True)
class Mech:
    label: str
    rd: tuple[str, ...]
    ry: tuple[str, ...]


def mechanisms():
    return [
        Mech("1ZD⊕2UD", ("U",), ("Z", "D", "RD")),
        Mech("1UD⊕2UD", ("U",), ("U", "D", "RD")),
        Mech("1UY⊕2UD", ("U",), ("U", "H", "RD")),
        Mech("1ZD+2ZD", ("D",), ("Z", "D")),
        Mech("1UD+2ZD", ("D",), ("U", "D")),
        Mech("1UY+2ZD", ("D",), ("U", "H")),
        Mech("1ZD⊕2Z", tuple(), ("Z", "D", "RD")),
        Mech("1UD⊕2Z", tuple(), ("U", "D", "RD")),
        Mech("1UY⊕2Z", tuple(), ("U", "H", "RD")),
        Mech("1Z⊕2ZD", ("D",), ("Z", "RD")),
        Mech("1D⊕2ZD", ("D",), ("D", "RD")),
        Mech("1Y⊕2ZD", ("D",), ("H", "RD")),
        Mech("1U⊕2ZU", ("U",), ("U", "RD")),
    ]


# EM fit of the CACE for one missingness mechanism.
# The outcome follows a two-part model: a logistic model for whether
# earnings are positive (H = 1(Y>0)) and a Gamma model (log link, shape nu) for the positive
# amount. The response indicators R^D and R^Y each have a logistic model whose extra
# covariates are set by the mechanism. All parts are fit jointly by EM over the latent (U, H).
class GammaEM:
    states = ((0, 0), (0, 1), (1, 0), (1, 1))
    def __init__(self, df, X, mech, init=None):
        self.X, self.m = np.asarray(X, float), mech
        self.Z = df.Z.to_numpy(int); self.Dobs = df.D.to_numpy(float)
        self.Y = df.Y.to_numpy(float); self.RD = df.RD.to_numpy(int)
        self.RY = df.RY.to_numpy(int); self.Hobs = df.H.to_numpy(int)
        self.n, self.p = self.X.shape
        self.outX = {u: np.column_stack([self.X, np.full(self.n, u), self.Z * u]) for u in (0, 1)}
        self.rdX = {u: self.design(u, 0, mech.rd) for u in (0, 1)}
        self.ryX = {(u, h): self.design(u, h, mech.ry) for u in (0, 1) for h in (0, 1)}
        self.par = self.start() if init is None else {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in init.items()}

    # Design matrix: covariates X plus the extra predictors this response model depends on.
    def design(self, u, h, names):
        vals = {"Z": self.Z.astype(float), "U": np.full(self.n, u, float),
                "D": (self.Z * u).astype(float), "H": np.full(self.n, h, float), "RD": self.RD.astype(float)}
        return np.column_stack([self.X] + [vals[a] for a in names]) if names else self.X

    # Initialize the parameters from simple complete-case fits.
    def start(self):
        par: dict[str, Any] = {}
        idx = (self.Z == 1) & (self.RD == 1)
        par["pi"] = fit_logit(self.X[idx], self.Dobs[idx], np.ones(idx.sum())) if idx.sum() > 10 else np.zeros(self.p)  # compliance: P(complier | X)
        idx = self.RY == 1
        bH = np.zeros(self.p + 2)
        if idx.sum() > 10: bH[: self.p] = fit_logit(self.X[idx], self.Hobs[idx], np.ones(idx.sum()))
        par["H"] = bH                                                                       # two-part model, part 1: P(Y>0 | X, U, D)
        idx = (self.RY == 1) & (self.Y > 0)
        bG = np.zeros(self.p + 2)
        bG[: self.p], nu = fit_gamma(self.X[idx], self.Y[idx], np.ones(idx.sum())) if idx.sum() > 10 else (np.zeros(self.p), 1.0)
        par["G"], par["nu"] = bG, nu                                                        # two-part model, part 2: Gamma mean (shape nu) for positive earnings
        z1 = self.Z == 1
        par["RD"] = fit_logit(np.vstack([self.rdX[0][z1], self.rdX[1][z1]]),
                              np.r_[self.RD[z1], self.RD[z1]], np.ones(2 * z1.sum()) / 2)    # treatment-response model R^D
        par["RY"] = fit_logit(np.vstack([self.ryX[u, h] for u, h in self.states]),
                              np.tile(self.RY, 4), np.ones(4 * self.n) / 4)                  # outcome-response model R^Y
        return par

    # Log-likelihood of one unit's observed data under latent state (U=u, H=h).
    def state_ll(self, u, h):
        d = self.Z * u
        pi_eta = np.clip(self.X @ self.par["pi"], -35, 35)
        ll = np.where(u == 1, lsig(pi_eta), l1sig(pi_eta))
        ok = ~((self.RD == 1) & (np.nan_to_num(self.Dobs, nan=-9).astype(int) != d))
        ok &= ~((self.Z == 0) & (self.RD == 0))
        z1 = self.Z == 1
        e = np.clip(self.rdX[u] @ self.par["RD"], -35, 35)
        ll[z1] += np.where(self.RD[z1] == 1, lsig(e[z1]), l1sig(e[z1]))
        e = np.clip(self.outX[u] @ self.par["H"], -35, 35)
        ll += np.where(h == 1, lsig(e), l1sig(e))
        ok &= ~((self.RY == 1) & (self.Hobs != h))
        if h == 1:
            pos = (self.RY == 1) & (self.Y > 0)
            if pos.any():
                nu = float(self.par["nu"])
                mu = np.exp(np.clip(self.outX[u] @ self.par["G"], -30, 30))[pos]
                y = self.Y[pos]
                ll[pos] += nu * np.log(nu) - gammaln(nu) + (nu - 1) * np.log(y) - nu * np.log(mu) - nu * y / mu
        e = np.clip(self.ryX[u, h] @ self.par["RY"], -35, 35)
        ll += np.where(self.RY == 1, lsig(e), l1sig(e))
        return np.where(ok, ll, -np.inf)

    # E-step: posterior weights over the four latent (U, H) states, and the log-likelihood.
    def e_step(self):
        M = np.vstack([self.state_ll(u, h) for u, h in self.states])
        den = logsumexp(M, axis=0)
        return np.exp(M - den), float(np.sum(den))

    # M-step: refit each model piece, weighting observations by the E-step posteriors.
    def m_step(self, W):
        old = self.par
        q0, q1 = W[0] + W[1], W[2] + W[3]
        self.par["pi"] = fit_logit(self.X, q1, np.ones(self.n), old["pi"])
        y0 = np.divide(W[1], q0, out=np.zeros(self.n), where=q0 > 1e-12)
        y1 = np.divide(W[3], q1, out=np.zeros(self.n), where=q1 > 1e-12)
        self.par["H"] = fit_logit(np.vstack([self.outX[0], self.outX[1]]), np.r_[y0, y1], np.r_[q0, q1], old["H"])
        pos = (self.RY == 1) & (self.Y > 0)
        self.par["G"], self.par["nu"] = fit_gamma(np.vstack([self.outX[0][pos], self.outX[1][pos]]),
                                                   np.r_[self.Y[pos], self.Y[pos]], np.r_[W[1, pos], W[3, pos]],
                                                   old["G"], math.log(float(old["nu"])))
        z1 = self.Z == 1
        self.par["RD"] = fit_logit(np.vstack([self.rdX[0][z1], self.rdX[1][z1]]), np.r_[self.RD[z1], self.RD[z1]],
                                   np.r_[q0[z1], q1[z1]], old["RD"])
        self.par["RY"] = fit_logit(np.vstack([self.ryX[u, h] for u, h in self.states]), np.tile(self.RY, 4), W.ravel(), old["RY"])

    # Run EM to convergence (or maxit iterations).
    def fit(self, maxit=100, tol=1e-6):
        prev = -np.inf
        for _ in range(maxit):
            W, _ = self.e_step(); self.m_step(W); _, ll = self.e_step()
            if np.isfinite(prev) and abs(ll - prev) <= tol * (1 + abs(prev)): break
            prev = ll
        return self

    # CACE: complier-weighted average of E[Y | D=1] - E[Y | D=0].
    def cace(self):
        pi = expit(np.clip(self.X @ self.par["pi"], -35, 35))
        def EY(d):
            Xo = np.column_stack([self.X, np.ones(self.n), np.full(self.n, d)])
            return expit(np.clip(Xo @ self.par["H"], -35, 35)) * np.exp(np.clip(Xo @ self.par["G"], -30, 30))
        return float(np.sum(pi * (EY(1) - EY(0))) / np.sum(pi))


# Fit every mechanism; return its CACE estimate and fitted parameters.
def fit_all(df, X, ms, maxit, init=None):
    out = {}
    for m in ms:
        fit = GammaEM(df, X, m, None if init is None else init.get(m.label)).fit(maxit=maxit)
        out[m.label] = (fit.cace(), fit.par)
    return out


# Nonparametric bootstrap, warm-starting each resample from the full-sample estimates.
_BOOT = {}

def boot_init(df, X, ms, params, maxit): _BOOT.update(df=df, X=X, ms=ms, params=params, maxit=maxit)

def boot_one(task):
    b, seed = task
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(_BOOT["df"]), len(_BOOT["df"]))
    dfb, Xb = _BOOT["df"].iloc[idx].reset_index(drop=True), _BOOT["X"][idx]
    vals = []
    for m in _BOOT["ms"]:
        try:
            vals.append(GammaEM(dfb, Xb, m, _BOOT["params"].get(m.label)).fit(maxit=_BOOT["maxit"]).cace())
        except Exception:
            vals.append(np.nan)
    return b, vals


def boot_chunk(chunk): return [boot_one(task) for task in chunk]


def bootstrap(df, X, ms, params, B, seed, maxit, n_jobs):
    if B <= 0: return np.empty((0, len(ms)))
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(B)]
    tasks = [(i, seeds[i]) for i in range(B)]
    ans = np.full((B, len(ms)), np.nan)
    n_jobs = max(1, int(n_jobs)); done = 0
    if n_jobs == 1:
        boot_init(df, X, ms, params, maxit)
        for task in tasks:
            b, vals = boot_one(task); ans[b, :] = vals; done += 1
        return ans
    for start in range(0, B, 50):
        block = tasks[start:start + 50]
        chunks = [block[i:i + 3] for i in range(0, len(block), 3)]
        with mp.Pool(n_jobs, initializer=boot_init, initargs=(df, X, ms, params, maxit), maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(boot_chunk, chunks):
                for b, vals in res:
                    ans[b, :] = vals; done += 1
    return ans


# Manuscript outputs: one figure and one small TeX file with the numbers used in the text.
def mech_tex(s):
    s = s.replace("⊕", r"\oplus ")
    s = re.sub(r"([12])([A-Z]+)", r"\1\\mathrm{\2}", s)
    return "$" + s + "$"


# Forest plot of the per-mechanism CACE point estimates with bootstrap confidence intervals.
def write_forest(path, ms, point, boot):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    labels = [mech_tex(m.label) for m in ms]
    lo = np.nanpercentile(boot, 2.5, axis=0) if boot.size else point.copy()
    hi = np.nanpercentile(boot, 97.5, axis=0) if boot.size else point.copy()
    y = np.arange(len(labels))[::-1]
    fig = plt.figure(figsize=(8.4, max(5.0, 0.39 * len(labels) + 0.8)), dpi=450)
    gs = GridSpec(1, 2, width_ratios=[5.3, 1.35], wspace=0.02, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=ax)
    for a in (ax, axr):
        for j, yy in enumerate(y):
            if j % 2 == 0: a.axhspan(yy - 0.47, yy + 0.47, color="#f7f7f7", zorder=0)
    if boot.size:
        ax.errorbar(point, y, xerr=np.vstack([point - lo, hi - point]), fmt="o", ms=4.7, mew=0,
                    elinewidth=1.45, capsize=3.1, capthick=1.45, color="#173f67", ecolor="#315f89", zorder=3)
    else:
        ax.plot(point, y, "o", ms=4.7, color="#173f67", zorder=3)
    ax.axvline(0, color="#777777", lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("CACE in weekly earnings (dollars)", fontsize=11)
    xmin = min(0.0, float(np.nanmin(lo)) - 1.5)
    xmax = max(50.0, float(np.nanmax(hi)) + 1.5)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(10 * math.floor(xmin / 10), 51, 10))
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.grid(axis="x", color="#d6d6d6", lw=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=5); ax.tick_params(axis="x", labelsize=9)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    axr.set_xlim(0, 1); axr.set_xticks([]); axr.set_yticks(y); axr.tick_params(left=False, labelleft=False)
    for sp in ("top", "right", "left", "bottom"): axr.spines[sp].set_visible(False)
    axr.text(0.98, len(labels) - 0.35, "Estimate (95% CI)", ha="right", va="bottom", fontsize=9.2, color="#333333")
    for yy, pnt, a, b in zip(y, point, lo, hi):
        txt = f"{pnt:.1f}" if not boot.size else f"{pnt:.1f} ({a:.1f}, {b:.1f})"
        axr.text(0.98, yy, txt, ha="right", va="center", fontsize=8.7, color="#333333")
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.11, top=0.985)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_numbers(path, df):
    z1 = df.Z.to_numpy() == 1
    d = df.D.to_numpy()
    y = df.Y.to_numpy()
    n = len(df)
    lines = {
        "njcsTotalN": f"{n:,}",
        "njcsAssignedJobCorpsN": f"{int(z1.sum()):,}",
        "njcsKnownNotEnrollPct": f"{100 * np.sum(z1 & (df.RD.to_numpy() == 1) & (d == 0)) / z1.sum():.1f}\\%",
        "njcsEnrollmentMissingPct": f"{100 * np.sum(z1 & (df.RD.to_numpy() == 0)) / z1.sum():.1f}\\%",
        "njcsOutcomeMissingPct": f"{100 * np.sum(df.RY.to_numpy() == 0) / n:.1f}\\%",
        "njcsZeroEarningsPct": f"{100 * np.sum((df.RY.to_numpy() == 1) & (y == 0)) / n:.1f}\\%",
    }
    with open(path, "w", encoding="utf-8") as f:
        for k, v in lines.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    p = argparse.ArgumentParser(description="NJCS empirical illustration for IV MNAR CACE identification.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--B", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--point-maxit", type=int, default=100)
    p.add_argument("--boot-maxit", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, X = load_njcs(args.data_dir)
    ms = mechanisms()
    fits = fit_all(df, X, ms, args.point_maxit)
    params = {k: par for k, (_, par) in fits.items()}
    point = np.array([fits[m.label][0] for m in ms])
    boot = bootstrap(df, X, ms, params, args.B, args.seed, args.boot_maxit, args.n_jobs)

    write_forest(os.path.join(args.out_dir, "njcs_cace_forest.png"), ms, point, boot)
    write_numbers(os.path.join(args.out_dir, "njcs_manuscript_numbers.tex"), df)

if __name__ == "__main__": main()
