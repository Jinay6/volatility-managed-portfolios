"""Volatility-managed portfolio construction (Moreira & Muir 2017, eq. 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DAYS_PER_MONTH = 15


def realised_variance(daily_returns: pd.Series, min_days: int = MIN_DAYS_PER_MONTH) -> pd.Series:
    """RV_t: sum of squared daily returns within calendar month t (month-end index).

    Months with fewer than ``min_days`` observations are dropped.
    """
    r = daily_returns.dropna()
    grouped = (r**2).groupby(r.index.to_period("M"))
    rv = grouped.sum()[grouped.count() >= min_days]
    rv.index = rv.index.to_timestamp("M")
    return rv.rename("RV")


def volatility_managed(factor_monthly: pd.Series, rv_monthly: pd.Series) -> tuple[pd.Series, float]:
    """Build f^sigma_{t+1} = (c / RV_t) * f_{t+1}.

    RV is lagged one month on its own index, so month t+1's position uses only
    information available at the end of month t. The constant c sets
    var(f^sigma) = var(f) over the sample. Note: c uses the full sample
    (the paper's convention); it rescales the strategy but does not affect
    Sharpe ratios or alpha t-statistics.
    """
    rv_lag = rv_monthly.shift(1).rename("RV_lag")
    df = pd.concat([factor_monthly.rename("f"), rv_lag], axis=1).dropna()
    raw = df["f"] / df["RV_lag"]
    c = float(np.sqrt(df["f"].var() / raw.var()))
    return (c * raw).rename("f_vm"), c
