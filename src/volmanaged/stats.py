"""Performance statistics and regressions (Newey-West HAC inference)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

MONTHS = 12
SQRT_MONTHS = np.sqrt(MONTHS)
NW_LAGS = 6


@dataclass(frozen=True)
class FactorStats:
    name: str
    n_months: int
    mean_orig: float
    mean_vm: float
    sd_orig: float
    sd_vm: float
    sharpe_orig: float
    sharpe_vm: float
    alpha_ann: float  # % per year
    alpha_t: float
    beta: float
    r2: float
    appraisal: float  # alpha / sigma(eps), annualised
    c_scale: float


def alpha_regression(managed: pd.Series, original: pd.Series) -> tuple[float, float, float, float, float]:
    """f^sigma = alpha + beta f + e. Returns monthly alpha, NW t(alpha), beta, R^2, resid sd."""
    df = pd.concat([managed.rename("y"), original.rename("x")], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return (
        float(res.params["const"]),
        float(res.tvalues["const"]),
        float(res.params["x"]),
        float(res.rsquared),
        float(np.std(res.resid, ddof=1)),
    )


def compute_stats(name: str, original: pd.Series, managed: pd.Series, c: float) -> FactorStats:
    df = pd.concat([original.rename("o"), managed.rename("m")], axis=1).dropna()
    mo, mm = df["o"].mean(), df["m"].mean()
    so, sm_ = df["o"].std(ddof=1), df["m"].std(ddof=1)
    alpha, t_alpha, beta, r2, resid_sd = alpha_regression(df["m"], df["o"])
    return FactorStats(
        name=name,
        n_months=len(df),
        mean_orig=mo,
        mean_vm=mm,
        sd_orig=so,
        sd_vm=sm_,
        sharpe_orig=mo / so * SQRT_MONTHS,
        sharpe_vm=mm / sm_ * SQRT_MONTHS,
        alpha_ann=alpha * MONTHS * 100,
        alpha_t=t_alpha,
        beta=beta,
        r2=r2,
        appraisal=alpha / resid_sd * SQRT_MONTHS,
        c_scale=c,
    )


def predictability_regression(returns: pd.Series, rv: pd.Series) -> tuple[float, float, float, int]:
    """r_{t+1} = a + b RV_t + e. Returns b (monthly, decimal), NW t(b), R^2, n."""
    df = pd.concat([returns.rename("r"), rv.shift(1).rename("rv_lag")], axis=1).dropna()
    res = sm.OLS(df["r"], sm.add_constant(df["rv_lag"])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return float(res.params["rv_lag"]), float(res.tvalues["rv_lag"]), float(res.rsquared), len(df)


def spanning_regression(managed: pd.Series, factors: dict[str, pd.Series]) -> tuple[float, float, float, int]:
    """Regress a managed factor on all original factors jointly. Returns alpha (%/yr), NW t, R^2, n."""
    df = pd.concat([managed.rename("y"), pd.DataFrame(factors)], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan, np.nan, 0
    res = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return float(res.params["const"] * MONTHS * 100), float(res.tvalues["const"]), float(res.rsquared), len(df)


def tangency_sharpe(returns: pd.DataFrame) -> float:
    """In-sample tangency Sharpe ratio, annualised: sqrt(mu' Sigma^-1 mu). In-sample, so optimistic."""
    df = returns.dropna()
    if len(df) < 60 or df.shape[1] < 2:
        return np.nan
    mu, sigma = df.mean().to_numpy(), df.cov().to_numpy()
    sr2 = mu @ np.linalg.pinv(sigma) @ mu
    return float(np.sqrt(max(sr2, 0.0)) * SQRT_MONTHS)
