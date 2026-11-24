"""Figures. Each function writes one PNG."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from .stats import FactorStats  # noqa: E402

PRIMARY, SECONDARY, ACCENT = "#0F766E", "#9CA3AF", "#B45309"
plt.rcParams.update(
    {
        "figure.figsize": (8, 4.5),
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#94A3B8",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)


def _save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _grouped_bars(
    ax: plt.Axes,
    names: Sequence[str],
    a: Sequence[float],
    b: Sequence[float],
    labels: tuple[str, str],
    colors: tuple[str, str],
) -> None:
    x, w = np.arange(len(names)), 0.38
    ax.bar(x - w / 2, a, w, color=colors[0], label=labels[0])
    ax.bar(x + w / 2, b, w, color=colors[1], label=labels[1])
    ax.set_xticks(x, names)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False)


def cumret_grid(records: Sequence[tuple[str, pd.Series, pd.Series]], out: Path) -> None:
    rows = (len(records) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.0 * rows))
    axes = np.atleast_2d(axes)
    for ax, (name, orig, vm) in zip(axes.flat, records, strict=False):
        cum = (1 + pd.concat([orig.rename("o"), vm.rename("v")], axis=1).dropna()).cumprod()
        ax.plot(cum.index, cum["o"], color=SECONDARY, linewidth=1.3, label="Buy and hold")
        ax.plot(cum.index, cum["v"], color=PRIMARY, linewidth=1.6, label="Vol-managed")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
    for ax in axes.flat[len(records) :]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("$1 invested, log scale: vol-managed vs buy-and-hold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def sharpe_compare(stats: Sequence[FactorStats], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _grouped_bars(
        ax,
        [s.name for s in stats],
        [s.sharpe_orig for s in stats],
        [s.sharpe_vm for s in stats],
        ("Original", "Vol-managed"),
        (SECONDARY, PRIMARY),
    )
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title(title)
    _save(fig, out)


def alpha_bars(stats: Sequence[FactorStats], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar([s.name for s in stats], [s.alpha_ann for s in stats], color=PRIMARY)
    ax.axhline(0, color="black", linewidth=0.6)
    for bar, s in zip(bars, stats, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s.alpha_ann + (0.2 if s.alpha_ann >= 0 else -0.5),
            f"{s.alpha_ann:.1f}%\n(t={s.alpha_t:.1f})",
            ha="center",
            va="bottom" if s.alpha_ann >= 0 else "top",
            fontsize=8,
        )
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title(title)
    _save(fig, out)


def paper_vs_full(paper: Sequence[FactorStats], full: Sequence[FactorStats], out: Path) -> None:
    by_name = {s.name: s.alpha_ann for s in full}
    names = [s.name for s in paper]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    _grouped_bars(
        ax,
        names,
        [s.alpha_ann for s in paper],
        [by_name.get(n, np.nan) for n in names],
        ("Paper sample (to 2015)", "Extended sample"),
        (PRIMARY, ACCENT),
    )
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title("Alpha in the original vs extended sample")
    _save(fig, out)


def subsample_alpha(per_window: dict[str, dict[str, float]], out: Path) -> None:
    factors = sorted({f for d in per_window.values() for f in d})
    x, w = np.arange(len(factors)), 0.8 / len(per_window)
    palette = [PRIMARY, "#14B8A6", "#5EEAD4", ACCENT]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (label, vals) in enumerate(per_window.items()):
        ax.bar(
            x + (i - (len(per_window) - 1) / 2) * w,
            [vals.get(f, np.nan) for f in factors],
            w,
            color=palette[i % len(palette)],
            label=label,
        )
    ax.set_xticks(x, factors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title("Alpha by subsample (non-overlapping windows)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out)


def predictability(rows: Sequence[dict], out: Path) -> None:
    ts = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.bar([r["Factor"] for r in rows], ts, color=[PRIMARY if abs(t) >= 1.96 else SECONDARY for t in ts])
    for y in (1.96, -1.96):
        ax.axhline(y, color=ACCENT, linewidth=1, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Newey-West t-statistic")
    ax.set_title("Does last month's variance predict next month's return?")
    _save(fig, out)


def spanning(rows: Sequence[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    _grouped_bars(
        ax,
        [r["Factor"] for r in rows],
        [r["alpha_bivariate"] for r in rows],
        [r["alpha_spanning"] for r in rows],
        ("Alpha vs own factor", "Alpha vs FF5 + MOM"),
        (SECONDARY, PRIMARY),
    )
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title("Spanning test, same sample (1963+): own-factor alpha vs alpha after FF5 + MOM")
    _save(fig, out)


def vol_vs_return(rv: pd.Series, monthly: pd.Series, out: Path) -> None:
    df = pd.concat([rv.rename("rv"), monthly.shift(-1).rename("ret")], axis=1).dropna()
    df["vol"] = np.sqrt(df["rv"] * 12)
    res = sm.OLS(df["ret"], sm.add_constant(df["vol"])).fit()
    xs = np.linspace(df["vol"].min(), df["vol"].max(), 100)
    fig, ax = plt.subplots()
    ax.scatter(df["vol"] * 100, df["ret"] * 100, s=8, color=PRIMARY, alpha=0.4)
    ax.plot(
        xs * 100,
        (res.params["const"] + res.params["vol"] * xs) * 100,
        color=ACCENT,
        linewidth=1.5,
        label=f"OLS slope = {res.params['vol']:.3f}",
    )
    ax.set_xlabel("Realised volatility, month t (annualised, %)")
    ax.set_ylabel("Market excess return, month t+1 (%)")
    ax.set_title("High volatility does not predict proportionally higher returns")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out)
