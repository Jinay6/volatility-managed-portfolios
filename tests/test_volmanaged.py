import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from volmanaged.data import parse_french_csv
from volmanaged.stats import alpha_regression, tangency_sharpe
from volmanaged.strategy import realised_variance, volatility_managed

rng = np.random.default_rng(42)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_test.CSV", text)
    return buf.getvalue()


def test_parse_monthly_keeps_first_block_and_converts_percent():
    text = (
        "Header text\n\n,Mkt-RF,SMB\n192607,   2.96,  -2.56\n192608,   2.64,  -1.17\n"
        "\n Annual Factors\n,Mkt-RF,SMB\n1927,  29.47, -2.04\n"
    )
    df = parse_french_csv(_zip(text), daily=False)
    assert list(df.columns) == ["Mkt-RF", "SMB"]
    assert len(df) == 2  # annual block excluded
    assert df.index[0] == pd.Timestamp("1926-07-31")  # month-end
    assert df.iloc[0, 0] == pytest.approx(0.0296)


def test_parse_daily():
    text = "Header\n,Mkt-RF\n19260701,0.10\n19260702,0.45\n\nCopyright\n"
    df = parse_french_csv(_zip(text), daily=True)
    assert len(df) == 2 and df.index[1] == pd.Timestamp("1926-07-02")
    assert df.iloc[1, 0] == pytest.approx(0.0045)


def _daily(months: int = 60) -> pd.Series:
    idx = pd.bdate_range("2000-01-01", periods=months * 21)
    return pd.Series(rng.normal(0, 0.01, len(idx)), idx)


def test_realised_variance_is_sum_of_squares_and_drops_thin_months():
    d = _daily(3)
    rv = realised_variance(d)
    jan = d[d.index.month == 1]
    assert rv.iloc[0] == pytest.approx((jan**2).sum())
    assert realised_variance(d.iloc[:10]).empty  # < 15 days -> dropped


def test_volatility_managed_uses_lagged_variance_and_matches_variance():
    d = _daily(120)
    rv = realised_variance(d)
    f = pd.Series(rng.normal(0.005, 0.04, len(rv)), rv.index)
    vm, c = volatility_managed(f, rv)
    # first month is lost to the lag; month t+1 is scaled by RV_t (no look-ahead)
    assert vm.index[0] == rv.index[1]
    t1 = rv.index[5]
    assert vm[t1] == pytest.approx(c * f[t1] / rv.iloc[4])
    assert vm.var() == pytest.approx(f.loc[vm.index].var(), rel=1e-9)


def test_alpha_regression_recovers_known_alpha():
    x = pd.Series(rng.normal(0, 0.04, 2000))
    y = 0.002 + 0.6 * x + rng.normal(0, 0.01, 2000)
    alpha, t, beta, r2, _ = alpha_regression(y, x)
    assert alpha == pytest.approx(0.002, abs=5e-4)
    assert beta == pytest.approx(0.6, abs=0.02)
    assert t > 5


def test_tangency_sharpe_single_asset_equivalent():
    a = pd.Series(rng.normal(0.01, 0.05, 600))
    b = pd.Series(rng.normal(0.0, 0.05, 600))
    sr = tangency_sharpe(pd.DataFrame({"a": a, "b": b}))
    assert sr >= a.mean() / a.std() * np.sqrt(12) - 1e-9  # combining can only weakly improve in-sample SR
