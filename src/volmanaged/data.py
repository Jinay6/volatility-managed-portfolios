"""Download, cache and parse factor returns from the Kenneth R. French Data Library."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FILES = {
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ff3_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
}


def fetch_zip(filename: str, cache_dir: Path) -> bytes:
    """Return the bytes of a French-library zip, downloading it once into ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / filename
    if cached.exists():
        return cached.read_bytes()
    print(f"  downloading {filename} ...")
    response = requests.get(BASE_URL + filename, timeout=60)
    response.raise_for_status()
    cached.write_bytes(response.content)
    return response.content


def parse_french_csv(zip_bytes: bytes, daily: bool) -> pd.DataFrame:
    """Parse the first data block of a French-library CSV.

    The files start with a text header, then the daily (YYYYMMDD) or monthly
    (YYYYMM) block, then an annual block. Only the first block is kept.
    Returns DECIMAL returns (the source is in percent) indexed by date
    (month-end for monthly data).
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        lines = z.read(name).decode("latin-1").splitlines()

    start = next(i for i, ln in enumerate(lines) if ln[:1].isdigit())
    width = 8 if daily else 6
    end = len(lines)
    for i in range(start, len(lines)):
        first = lines[i].strip().split(",")[0].strip()
        if not (first.isdigit() and len(first) == width):
            end = i
            break

    df = pd.read_csv(io.StringIO("\n".join([lines[start - 1], *lines[start:end]])))
    date_col = df.columns[0]
    dates = df[date_col].astype(str)
    if daily:
        df[date_col] = pd.to_datetime(dates, format="%Y%m%d")
    else:
        df[date_col] = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index(date_col).sort_index().astype(float) / 100.0
    df.columns = [c.strip() for c in df.columns]
    df.index.name = "date"
    return df


def load_factors(cache_dir: Path) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load monthly and daily excess returns for MKT, HML, SMB, MOM, RMW, CMA."""
    p = {k: parse_french_csv(fetch_zip(f, cache_dir), daily=k.endswith("daily")) for k, f in FILES.items()}

    def mom(df: pd.DataFrame) -> pd.Series:
        return df["Mom"] if "Mom" in df.columns else df.iloc[:, 0]

    daily = {
        "MKT": p["ff3_daily"]["Mkt-RF"],
        "HML": p["ff3_daily"]["HML"],
        "SMB": p["ff3_daily"]["SMB"],
        "MOM": mom(p["mom_daily"]),
        "RMW": p["ff5_daily"]["RMW"],
        "CMA": p["ff5_daily"]["CMA"],
    }
    monthly = {
        "MKT": p["ff3_monthly"]["Mkt-RF"],
        "HML": p["ff3_monthly"]["HML"],
        "SMB": p["ff3_monthly"]["SMB"],
        "MOM": mom(p["mom_monthly"]),
        "RMW": p["ff5_monthly"]["RMW"],
        "CMA": p["ff5_monthly"]["CMA"],
    }
    return monthly, daily
