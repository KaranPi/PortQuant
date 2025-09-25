# src/quantlib/io.py


#Preferences baked in:
# -Pandas 1.2-friendly: no newer kwargs.
# -NaN policy: keep pre-IPO/non-trading as NaN (no fill here).
# -Safety: non-positive closes are nulled to NaN (bad ticks).
# -Anchor left-join: avoids “empty intersection” when newer IPOs exist.


#Function: io.py is your data plumbing. It takes messy raw CSVs and outputs clean, consistent, reproducible Close panels and coverage reports that every other step depends on.



from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

# ------------------------- Filesystem helpers -------------------------

def ensure_dir(p: Path) -> None:
    """Create parent directory for a file path or the directory itself."""
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

def save_artifact(obj, path: Path, kind: str = "pkl") -> Path:
    """
    Save DataFrame/Series to pkl or csv. Returns the path.
    kind: 'pkl' or 'csv'
    """
    path = Path(path)
    ensure_dir(path)
    if kind.lower() == "pkl":
        obj.to_pickle(path)
    elif kind.lower() == "csv":
        obj.to_csv(path, index=True)
    else:
        raise ValueError("kind must be 'pkl' or 'csv'")
    return path

# ------------------------- CSV loading -------------------------

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    """Convert to float; non-convertible → NaN."""
    return pd.to_numeric(s, errors="coerce")

def load_one_csv(
    csv_path: str,
    rename_map: Dict[str, str],
    series_filter: Optional[Iterable[str]] = ("EQ",),
) -> pd.DataFrame:
    """
    Load one raw NSE-style CSV and standardize:
      - parse DATE → index
      - rename columns (e.g., 'CLOSE'→'close')
      - keep SERIES in series_filter if present
      - drop rows with missing date/symbol/close
      - sort by date, drop duplicate (date,symbol)
      - return indexed by 'date'
    """
    csv_path = str(csv_path)
    # Read header to intersect available columns
    cols = pd.read_csv(csv_path, nrows=0).columns
    usecols = [c for c in rename_map.keys() if c in cols]

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=["DATE"] if "DATE" in usecols else None,
        infer_datetime_format=True,
        engine="c",
        low_memory=False,
    ).rename(columns=rename_map)

    # Optional SERIES filter
    if series_filter and "series" in df.columns:
        df = df[df["series"].isin(set(series_filter))]

    # Keep only valid rows
    need = [c for c in ["date", "symbol", "close"] if c in df.columns]
    df = df.dropna(subset=need)

    # Normalize types
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("category")
    if "close" in df.columns:
        df["close"] = _to_numeric_safe(df["close"])

    # Sort & dedupe
    if "date" in df.columns:
        df = df.sort_values("date").drop_duplicates(subset=["date", "symbol"])

    return df.set_index("date")

def load_folder(
    data_raw_dir: str,
    rename_map: Dict[str, str],
    series_filter: Optional[Iterable[str]] = ("EQ",),
    glob: str = "*.csv",
) -> Dict[str, pd.DataFrame]:
    """
    Load all CSVs in a folder into a dict: {symbol -> OHLCV dataframe (indexed by date)}.
    Assumes each CSV contains one symbol; extracts symbol from 'symbol' column (preferred)
    or from filename stem if necessary.
    """
    per_symbol: Dict[str, pd.DataFrame] = {}
    for fp in sorted(Path(data_raw_dir).glob(glob)):
        dfi = load_one_csv(str(fp), rename_map, series_filter)
        if dfi.empty:
            continue
        if "symbol" in dfi.columns:
            for sym, g in dfi.groupby("symbol"):
                per_symbol[str(sym)] = g.copy()
        else:
            # fallback: use filename stem as symbol
            sym = fp.stem.upper()
            per_symbol[sym] = dfi.copy().assign(symbol=sym)
    return per_symbol

# ------------------------- Coverage & anchor join -------------------------

def coverage_table(per_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Report per-symbol coverage: rows, start, end. Sorted by rows desc.
    """
    rows = []
    for sym, df in per_symbol.items():
        if df.empty:
            rows.append((sym, 0, None, None))
        else:
            idx = df.index
            rows.append((sym, len(df), idx.min().date(), idx.max().date()))
    cov = pd.DataFrame(rows, columns=["symbol", "rows", "start", "end"])
    return cov.sort_values("rows", ascending=False).reset_index(drop=True)

def choose_anchor_symbol(cov: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """
    Choose an anchor symbol:
      - if 'preferred' appears in coverage, return it;
      - else pick the symbol with the most rows (longest history).
    """
    if preferred is not None:
        m = cov[cov["symbol"].str.upper() == preferred.upper()]
        if not m.empty:
            return str(m.iloc[0]["symbol"])
    return str(cov.iloc[0]["symbol"])

def build_close_panel_anchor_leftjoin(
    per_symbol: Dict[str, pd.DataFrame],
    anchor_symbol: str,
    drop_nonpositive: bool = True,
) -> pd.DataFrame:
    """
    Build a wide Close panel by LEFT-JOINING every symbol to the anchor's date index.
    - preserves pre-IPO NaNs for newer listings
    - keeps only anchor's trading days (no forward fill here)
    - optional: drop non-positive closes (bad ticks)
    Returns: DataFrame (index=anchor dates, columns=symbols, values=close)
    """
    if anchor_symbol not in per_symbol:
        raise KeyError(f"Anchor symbol '{anchor_symbol}' not found in per_symbol keys.")

    anchor_idx = per_symbol[anchor_symbol].index
    out = pd.DataFrame(index=anchor_idx)

    for sym, df in per_symbol.items():
        s = df["close"].rename(sym)
        if drop_nonpositive:
            s = s.where(s > 0.0, np.nan)
        # left-join onto anchor index
        out[s.name] = s.reindex(anchor_idx)

    # Sort columns for reproducibility
    out = out.sort_index().sort_index(axis=1)
    return out
