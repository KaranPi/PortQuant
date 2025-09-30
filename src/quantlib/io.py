# src/quantlib/io.py


#Preferences baked in:
# -Pandas 1.2-friendly: no newer kwargs.
# -NaN policy: keep pre-IPO/non-trading as NaN (no fill here).
# -Safety: non-positive closes are nulled to NaN (bad ticks).
# -Anchor left-join: avoids “empty intersection” when newer IPOs exist.


#Function: io.py is your data plumbing. It takes messy raw CSVs and outputs clean, consistent, reproducible Close panels and coverage reports that every other step depends on.



from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import time
import re

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


def inspect_index_csv(fp):
    """Quick peek: show columns & which ones would be picked."""
    df = pd.read_csv(fp)
    print("Columns:", list(df.columns))
    try:
        ndf = _normalise_index_df(df)
        print("Detected:", "DATE" in ndf, "CLOSE" in ndf, "SYMBOL" in ndf)
        print(ndf.head(3))
    except Exception as e:
        print("Normalizer error:", e)



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





# --- Jugaad-data helpers: stocks + index + run-dirs --------------------------

from datetime import date, datetime, timedelta
import time
import pandas as pd

try:
    from jugaad_data.nse import stock_df, index_df
except Exception:
    stock_df = None
    index_df = None

def ensure_run_dirs(root: Path, run_date_str: str):
    """
    Create and return (RUN_DIR, RAW, INT, FIG) inside: root/<DD-MM-YYYY>/
    """
    run_dir = Path(root) / run_date_str
    raw = run_dir / "data_raw"
    dint = run_dir / "data_int"
    fig = run_dir / "figures"
    for p in (raw, dint, fig):
        p.mkdir(parents=True, exist_ok=True)
    return run_dir, raw, dint, fig

def fetch_stocks_to_csv(
    symbols,
    from_date: date,
    to_date: date,
    out_dir: Path,
    series: str = "EQ",
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch a basket of NSE symbols to CSV (one file per symbol).
    Returns a long-form DataFrame with DATE,SYMBOL,CLOSE (only rows we need downstream).
    Safe with your current pandas version.
    """
    if stock_df is None:
        raise ImportError("jugaad-data is not installed in this environment.")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    ok, bad = 0, 0
    for sym in symbols:
        try:
            df = stock_df(symbol=sym, from_date=from_date, to_date=to_date, series=series)
            if df is None or df.empty:
                bad += 1
                continue
            # standardize + save raw
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"]).sort_values("DATE")
            df.to_csv(out_dir / f"{sym}.csv", index=False, date_format="%Y-%m-%d")
            # keep only what we need for downstream panel builds
            if "CLOSE" in df.columns:
                frames.append(df[["DATE"]].assign(SYMBOL=sym, CLOSE=pd.to_numeric(df["CLOSE"], errors="coerce")))
            ok += 1
            time.sleep(sleep_s)  # polite pacing
        except Exception:
            bad += 1
    if not frames:
        return pd.DataFrame(columns=["DATE","SYMBOL","CLOSE"])
    market = (pd.concat(frames, ignore_index=True)
                .dropna(subset=["DATE","SYMBOL"])
                .drop_duplicates(subset=["DATE","SYMBOL"])
                .sort_values(["DATE","SYMBOL"]))
    # convenience copy in the raw folder for quick inspection
    market.to_csv(out_dir / "market_data.csv", index=False, date_format="%Y-%m-%d")
    return market



    
def build_close_panel_from_raw(raw_dir: Path, symbols=None, anchor: str = "auto", positive_only: bool = True) -> pd.DataFrame:
    """
    Read per-symbol CSVs in raw_dir, build an anchor-left-join Close panel, and return it.
    If 'symbols' is provided, restrict to those files; otherwise use all '*.csv' except 'market_data.csv'.
    """
    raw_dir = Path(raw_dir)
    files = list(raw_dir.glob("*.csv"))
    files = [p for p in files if p.name.lower() != "market_data.csv"]
    if symbols:
        symset = set(map(str.upper, symbols))
        files = [p for p in files if p.stem.upper() in symset]
    if not files:
        return pd.DataFrame()

    per = {}
    for p in files:
        try:
            df = pd.read_csv(p, parse_dates=["DATE"])
            if "CLOSE" not in df.columns:
                continue
            dfx = (df[["DATE","CLOSE"]]
                    .rename(columns={"DATE":"date","CLOSE":"close"})
                    .dropna(subset=["date","close"])
                    .sort_values("date")
                    .set_index("date"))
            per[p.stem] = dfx
        except Exception:
            pass

    if not per:
        return pd.DataFrame()

    # anchor = longest series
    if anchor == "auto":
        anchor_sym = max(per.items(), key=lambda kv: len(kv[1]))[0]
    else:
        anchor_sym = anchor if anchor in per else max(per.items(), key=lambda kv: len(kv[1]))[0]

    panel = per[anchor_sym][["close"]].rename(columns={"close": anchor_sym})
    for sym, dfx in per.items():
        if sym == anchor_sym:
            continue
        panel = panel.join(dfx["close"].rename(sym), how="left")

    if positive_only:
        panel = panel.where(panel > 0)

    return panel





# ---------------- Index CSV auto-loader ----------------


def _normalize_name(col: str) -> str:
    """
    Normalize a column name to a compact, comparable token:
    - lowercase
    - remove non-alphanumerics
    Example: 'Close Price' -> 'closeprice', 'Index Name' -> 'indexname'
    """
    return re.sub(r'[^a-z0-9]+', '', str(col).lower())

def _pick_column(df: pd.DataFrame, choices: list) -> str:
    """
    Find the first column whose normalized name matches any token in `choices`.
    Returns the original column name (not normalized).
    Raises KeyError if not found.
    """
    norm_map = { _normalize_name(c): c for c in df.columns }
    for token in choices:
        if token in norm_map:
            return norm_map[token]
    raise KeyError(f"None of {choices} found in columns={list(df.columns)}")

def _normalize_name(col: str) -> str:
    # normalize "Close Price" -> "closeprice", "TIMESTAMP" -> "timestamp"
    return re.sub(r'[^a-z0-9]+', '', str(col).lower())

def _pick_column(df: pd.DataFrame, choices: list) -> str:
    norm_map = { _normalize_name(c): c for c in df.columns }
    for token in choices:
        if token in norm_map:
            return norm_map[token]
    raise KeyError(f"None of {choices} found in columns={list(df.columns)}")

def _parse_date_robust(s: pd.Series) -> pd.Series:
    # try day-first first (NSE is commonly dd-mm-YYYY), then fallback
    x = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if x.isna().mean() > 0.5:
        x = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    # strip any timezone, then normalize to midnight
    try:
        x = x.dt.tz_localize(None)
    except Exception:
        try:
            x = x.dt.tz_convert(None)
        except Exception:
            pass
    return x.dt.normalize()

def _normalise_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize index CSV columns to DATE (datetime, normalized) and CLOSE (float).
    Accepts many variants seen in nselib/CSV dumps.
    Keeps SYMBOL/Index Name if present to name the series.
    """
    date_candidates  = ['date', 'historicaldate', 'timestamp', 'tradedate']
    close_candidates = [
        'close', 'closeprice', 'closingprice',
        'indexclose', 'indexclosevalue',
        'closeindexval', 'closeindexvalue'  # NSE variants (e.g., CLOSE_INDEX_VAL)
    ]
    name_candidates  = ['symbol', 'indexname', 'name', 'index']

    df = df.copy()

    try:
        date_col  = _pick_column(df, date_candidates)
        close_col = _pick_column(df, close_candidates)
    except KeyError:
        raise ValueError("Index CSV is missing usable DATE and/or CLOSE columns "
                         f"(got columns: {list(df.columns)})")

    try:
        name_col = _pick_column(df, name_candidates)
    except KeyError:
        name_col = None

    out = pd.DataFrame({
        "DATE": _parse_date_robust(df[date_col]),
        "CLOSE": pd.to_numeric(df[close_col], errors="coerce")
    })
    if name_col is not None:
        out["SYMBOL"] = df[name_col].astype(str).str.strip()

    # Drop NaT dates, drop dup dates (keep last), sort
    out = (out.dropna(subset=["DATE"])
              .drop_duplicates(subset=["DATE"], keep="last")
              .sort_values("DATE")
              .reset_index(drop=True))
    return out


def find_index_csvs(raw_dir) -> list:
    """
    Return sorted list of paths like .../data_raw/index_*.csv
    """
    raw = Path(raw_dir)
    return sorted(raw.glob("index_*.csv"))


def load_index_closes(raw_dir, prefer: list = None) -> dict:
    """
    Auto-load all index_*.csv in data_raw.
    Returns dict: { index_name -> pd.Series of CLOSE (DATE index) }

    prefer: optional list of names (e.g., ["NIFTY50", "NIFTYBANK"])
            used only by convenience selection helper below.
    """
    out = {}
    for fp in find_index_csvs(raw_dir):
        try:
            df = pd.read_csv(fp)
            df = _normalise_index_df(df)
            # Name: try CSV SYMBOL unique; else derive from filename
            if "SYMBOL" in df.columns and df["SYMBOL"].notna().any():
                sym = str(df["SYMBOL"].dropna().iloc[0]).strip()
            else:
                # index_NIFTY50.csv -> NIFTY50
                m = re.match(r"index_(.+)\.csv$", fp.name, flags=re.I)
                sym = m.group(1) if m else fp.stem.replace("index_", "")
            sym = sym.replace(" ", "").upper()  # canonical key
            s = df.set_index("DATE")["CLOSE"].astype(float).rename(sym)
            out[sym] = s
        except Exception as e:
            # keep quiet but informative
            print(f"[index loader] Skipped {fp.name}: {e}")
            continue
    return out


def select_index_series(index_map: dict, prefer: list = None) -> pd.Series:
    """
    Pick one index series from load_index_closes().
    prefer: ordered list of names to match after stripping spaces & upper,
            e.g. ["NIFTY50", "NIFTYBANK", "NIFTY500"].
    Falls back to first available if no match.
    Returns empty Series if none found.
    """
    if not index_map:
        return pd.Series(dtype=float)
    if prefer:
        pref_norm = [p.replace(" ", "").upper() for p in prefer]
        for p in pref_norm:
            if p in index_map:
                return index_map[p]
    # fallback: first
    k = next(iter(index_map.keys()))
    return index_map[k]


def attach_index_to_panel(prices_close: pd.DataFrame,
                          raw_dir,
                          prefer: list = None,
                          how: str = "left") -> (pd.DataFrame, str):
    """
    Join the selected index CLOSE series as an extra column to your prices panel.
    Returns (joined_panel, index_name). If no index found, returns (original, None).
    """
    idx_map = load_index_closes(raw_dir)
    idx = select_index_series(idx_map, prefer=prefer)
    if idx.empty:
        return prices_close, None
    name = idx.name
    joined = prices_close.join(idx.to_frame(), how=how)
    return joined, name


def load_index_and_merge(
    int_dir,
    raw_dir,
    prefer=None,                          # e.g. ["NIFTY50","NIFTYBANK"]
    panel_candidates=None,                # override panel filenames to look for
    out_basename=None,                    # custom output basename (without extension)
    how: str = "left",                    # join method for dates
    save_csv: bool = True                 # also write CSV alongside PKL
):
    """
    Reads a Close-price panel from `data_int`, appends a detected index Close
    (from `data_raw/index_*.csv`) as a new column, and writes `*_with_index.pkl/csv`.

    Returns dict with: {"panel_path","out_pkl","out_csv","index_name","shape"}.
    """
    int_dir = Path(int_dir); raw_dir = Path(raw_dir)
    if panel_candidates is None:
        # Most common names in our flow; first match wins
        panel_candidates = [
            "prices_close_anchor_leftjoin.pkl",
            "prices_close.pkl"
        ]

    panel_path = None
    for name in panel_candidates:
        p = int_dir / name
        if p.exists():
            panel_path = p
            break
    if panel_path is None:
        raise FileNotFoundError(
            f"No panel found in {int_dir}. Tried: {panel_candidates}"
        )

    # Load panel
    prices_close = pd.read_pickle(panel_path)

    # Attach index
    joined, idx_name = attach_index_to_panel(prices_close, raw_dir, prefer=prefer, how=how)

    # If index already present (same name), avoid duplicating
    if idx_name and idx_name in prices_close.columns:
        joined = prices_close  # it was already there

    # Decide output filenames
    stem = out_basename or (panel_path.stem + "_with_index")
    out_pkl = int_dir / f"{stem}.pkl"
    out_csv = int_dir / f"{stem}.csv"

    # Save
    joined.to_pickle(out_pkl)
    if save_csv:
        joined.to_csv(out_csv)

    meta_path = int_dir / "index_meta.txt"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(idx_name or "")
    except Exception:
        pass

    return {
        "panel_path": str(panel_path),
        "out_pkl": str(out_pkl),
        "out_csv": str(out_csv) if save_csv else None,
        "index_name": idx_name,
        "shape": tuple(joined.shape),
        "meta_path": str(meta_path),
    }






