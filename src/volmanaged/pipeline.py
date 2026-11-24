"""End-to-end replication: python -m volmanaged [--data-dir DIR] [--results-dir DIR]."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import plots
from .data import load_factors
from .stats import (
    MONTHS,
    SQRT_MONTHS,
    FactorStats,
    alpha_regression,
    compute_stats,
    predictability_regression,
    spanning_regression,
    tangency_sharpe,
)
from .strategy import realised_variance, volatility_managed

FACTORS = ["MKT", "HML", "MOM", "RMW", "CMA"]
PAPER_END = pd.Timestamp("2015-12-31")
# Non-overlapping subsample windows (inclusive bounds).
SUBSAMPLES = {
    "1926-1965": ("1926-01-01", "1965-12-31"),
    "1966-1995": ("1966-01-01", "1995-12-31"),
    "1996-2015": ("1996-01-01", "2015-12-31"),
    "2016-latest": ("2016-01-01", None),
}

Records = list[tuple[str, pd.Series, pd.Series]]


def _window(s: pd.Series, start: str | pd.Timestamp | None, end: str | pd.Timestamp | None) -> pd.Series:
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s


def run_sample(
    monthly: dict[str, pd.Series], daily: dict[str, pd.Series], factors: list[str], start=None, end=None
) -> tuple[list[FactorStats], Records, pd.DataFrame]:
    """Vol-manage each factor within [start, end] and compute its statistics."""
    stats, records = [], []
    for name in factors:
        f_m = _window(monthly[name], start, end)
        managed, c = volatility_managed(f_m, realised_variance(_window(daily[name], start, end)))
        f_m = f_m.loc[managed.index]
        if len(managed) < 36:
            continue
        stats.append(compute_stats(name, f_m, managed, c))
        records.append((name, f_m, managed))
    table = pd.DataFrame(
        [
            {
                "Factor": s.name,
                "N months": s.n_months,
                "Mean orig (%/yr)": s.mean_orig * MONTHS * 100,
                "Mean vm (%/yr)": s.mean_vm * MONTHS * 100,
                "Vol orig (%/yr)": s.sd_orig * SQRT_MONTHS * 100,
                "Vol vm (%/yr)": s.sd_vm * SQRT_MONTHS * 100,
                "Sharpe orig": s.sharpe_orig,
                "Sharpe vm": s.sharpe_vm,
                "Alpha (%/yr)": s.alpha_ann,
                "t(alpha)": s.alpha_t,
                "Beta": s.beta,
                "R^2": s.r2,
                "Appraisal ratio": s.appraisal,
            }
            for s in stats
        ]
    ).set_index("Factor")
    return stats, records, table


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="cache for French-library zips")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    tables, figs = args.results_dir / "tables", args.results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    print("Loading Kenneth French factor data ...")
    monthly, daily = load_factors(args.data_dir)
    last = max(s.index.max() for s in monthly.values()).strftime("%Y-%m")

    stats_paper, _, table_paper = run_sample(monthly, daily, FACTORS, end=PAPER_END)
    stats_full, records_full, table_full = run_sample(monthly, daily, FACTORS)
    table_paper.to_csv(tables / "results_paper_sample.csv")
    table_full.to_csv(tables / "results_full_sample.csv")

    per_window: dict[str, dict[str, float]] = {}
    rows = []
    for label, (s, e) in SUBSAMPLES.items():
        st, _, _ = run_sample(monthly, daily, FACTORS, s, e)
        per_window[label] = {x.name: x.alpha_ann for x in st}
        rows += [
            {
                "Subsample": label,
                "Factor": x.name,
                "N months": x.n_months,
                "Alpha (%/yr)": x.alpha_ann,
                "t(alpha)": x.alpha_t,
            }
            for x in st
        ]
    pd.DataFrame(rows).to_csv(tables / "subsample_alpha.csv", index=False)

    pred = []
    for name in FACTORS:
        b, t, r2, n = predictability_regression(monthly[name], realised_variance(daily[name]))
        pred.append({"Factor": name, "b (%/yr per unit RV)": b * MONTHS * 100, "t": t, "R^2": r2, "n": n})
    pd.DataFrame(pred).to_csv(tables / "predictability.csv", index=False)

    # Spanning test. Both alphas are estimated on the SAME months (those where all
    # five factors exist, i.e. from 1963), so the comparison is like-for-like.
    originals = {n: monthly[n] for n in FACTORS}
    span = []
    for name, _, vm in records_full:
        common = pd.concat([vm.rename("vm"), pd.DataFrame(originals)], axis=1).dropna().index
        a_bi, t_bi, *_ = alpha_regression(vm.loc[common], monthly[name].loc[common])
        a, t, r2, n = spanning_regression(vm.loc[common], originals)
        span.append(
            {
                "Factor": name,
                "alpha_bivariate": a_bi * MONTHS * 100,
                "t_bivariate": t_bi,
                "alpha_spanning": a,
                "t_spanning": t,
                "R^2": r2,
                "n": n,
                "start": common.min().strftime("%Y-%m"),
            }
        )
    pd.DataFrame(span).to_csv(tables / "spanning.csv", index=False)

    mkt = next(s for s in stats_full if s.name == "MKT")
    combo = pd.DataFrame(
        [
            {"Portfolio": "Market, buy and hold", "Sharpe (annualised)": mkt.sharpe_orig},
            {"Portfolio": "Market, vol-managed", "Sharpe (annualised)": mkt.sharpe_vm},
            {
                "Portfolio": "Tangency of original factors (in-sample)",
                "Sharpe (annualised)": tangency_sharpe(pd.concat(originals, axis=1)),
            },
            {
                "Portfolio": "Tangency of vol-managed factors (in-sample)",
                "Sharpe (annualised)": tangency_sharpe(pd.concat({n: v for n, _, v in records_full}, axis=1)),
            },
        ]
    )
    combo.to_csv(tables / "combination.csv", index=False)

    plots.cumret_grid([r for r in records_full if r[0] in {"MKT", "HML", "MOM", "RMW"}], figs / "cumret_grid.png")
    plots.sharpe_compare(stats_full, figs / "sharpe_full.png", f"Sharpe ratios, 1926-{last}")
    plots.alpha_bars(stats_full, figs / "alpha_full.png", f"Alpha of vol-managed factors (Newey-West), 1926-{last}")
    plots.paper_vs_full(stats_paper, stats_full, figs / "alpha_paper_vs_full.png")
    plots.subsample_alpha(per_window, figs / "alpha_subsamples.png")
    plots.predictability(pred, figs / "predictability.png")
    plots.spanning(span, figs / "spanning.png")
    plots.vol_vs_return(realised_variance(daily["MKT"]), monthly["MKT"], figs / "vol_vs_return.png")

    print("\nPaper sample (to 2015-12):\n", table_paper.round(2).to_string())
    print(f"\nFull sample (to {last}):\n", table_full.round(2).to_string())
    print(f"\nTables -> {tables}/   Figures -> {figs}/")


if __name__ == "__main__":
    main()
