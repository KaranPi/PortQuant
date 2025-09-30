#Notes on compatibility

#Avoids newer kwargs (works on pandas 1.2.x).

#Sets min_periods via min_frac to suppress future warnings and keep windows robust.

#kurt() is excess kurtosis (Normal ⇒ 0); that matches how we’ve been reading your numbers.

#Where this module is used + notebook changes
#New focused notebooks that call stats.py

# 03_stats.ipynb (rolling mean/std/z + skew/kurt)

# 05_corr_beta.ipynb (correlation & beta)

# ----------------------------------------------------


# src/quantlib/stats.py


import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Sequence, List

# --------------------------- small helpers ---------------------------

def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a clean DateTimeIndex (sorted, unique)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _minp(window: int, min_frac: float) -> int:
    """Compute integer min_periods for a rolling window."""
    return max(2, int(np.ceil(window * float(min_frac))))

# --------------------- rolling mean / std / z-score ------------------

def rolling_mean_std(
    df: pd.DataFrame, window: int, min_frac: float = 0.8, ddof: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling mean and std (daily units) for each column.
    NaN-safe: a window must have >= min_frac * window valid points.
    """
    df = _ensure_dtindex(df.astype(float))
    mp = _minp(window, min_frac)
    mu = df.rolling(window, min_periods=mp).mean()
    sd = df.rolling(window, min_periods=mp).std(ddof=ddof)
    return mu, sd

def z_scores(
    df: pd.DataFrame, window: int, min_frac: float = 0.8, ddof: int = 1
) -> pd.DataFrame:
    """
    Rolling z-score: (x - mean) / std.
    """
    mu, sd = rolling_mean_std(df, window, min_frac=min_frac, ddof=ddof)
    z = (df - mu) / sd
    return z

# ------------------------ rolling skew / kurtosis --------------------

def rolling_skew_kurt(
    df: pd.DataFrame, window: int, min_frac: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling skewness and kurtosis (excess) for each column.
    - skew(): Pearson’s moment skewness
    - kurt(): Fisher’s excess kurtosis (Normal => 0)
    """
    df = _ensure_dtindex(df.astype(float))
    mp = _minp(window, min_frac)
    skew = df.rolling(window, min_periods=mp).skew()
    kurt = df.rolling(window, min_periods=mp).kurt()
    return skew, kurt

# ------------------------- correlation matrices ---------------------

def corr_full(
    df: pd.DataFrame, min_obs: int = 126, method: str = "pearson"
) -> pd.DataFrame:
    """
    Full-sample pairwise correlation with a minimum overlap (min_obs).
    Uses pandas' pairwise NaN handling.
    """
    df = _ensure_dtindex(df.astype(float))
    # pandas 1.2 supports min_periods in .corr
    c = df.corr(method=method, min_periods=int(min_obs))
    # ensure diag = 1 where present
    for i in c.index:
        if i in c.columns:
            c.loc[i, i] = 1.0
    return c

def corr_rolling_mats(
    df: pd.DataFrame, window: int, min_frac: float = 0.8, method: str = "pearson"
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Rolling correlation *matrices*. Returns a dict: {end_date -> corr_matrix}.
    Note: This is memory expensive (one matrix per day). Use for selected windows.
    """
    df = _ensure_dtindex(df.astype(float))
    mp = _minp(window, min_frac)
    out: Dict[pd.Timestamp, pd.DataFrame] = {}
    # iterate end points; compute corr on each rolling block
    for i in range(len(df.index)):
        j0 = i - window + 1
        if j0 < 0:
            continue
        block = df.iloc[j0:i+1]
        # require sufficient non-NaN in each pair; pandas handles pairwise min_periods internally
        c = block.corr(method=method, min_periods=mp)
        # hard-set diagonal
        for s in c.index:
            if s in c.columns:
                c.loc[s, s] = 1.0
        out[df.index[i]] = c
    return out

# ------------------------------ beta --------------------------------

def beta_against_index(
    r_df: pd.DataFrame,
    index_col: str,
    window: Optional[int] = None,
    min_frac: float = 0.8
) -> pd.DataFrame:
    """
    β(stock ← index) via cov/var.
    - If window is None: full-sample beta (one-row DataFrame).
    - If window is provided: rolling β (rows=dates, cols=symbols).
    """
    r_df = _ensure_dtindex(r_df.astype(float))
    if index_col not in r_df.columns:
        raise KeyError(f"Index column '{index_col}' not found in returns DataFrame.")

    idx = r_df[index_col]

    if window is None:
        # full-sample: cov(stock, idx) / var(idx)
        var_i = idx.var(ddof=1)
        betas = {}
        for col in r_df.columns:
            if col == index_col:
                betas[col] = 1.0
            else:
                cov = r_df[[col, index_col]].cov().iloc[0, 1]
                betas[col] = cov / var_i if np.isfinite(var_i) and var_i > 0 else np.nan
        return pd.DataFrame([betas], index=["full_sample"])

    # rolling case
    mp = _minp(window, min_frac)
    out = pd.DataFrame(index=r_df.index, columns=r_df.columns, dtype=float)

    # Precompute rolling var of index
    var_i = idx.rolling(window, min_periods=mp).var(ddof=1)

    for col in r_df.columns:
        if col == index_col:
            out[col] = 1.0
            continue
        # pairwise rolling covariance with index:
        cov_si = r_df[[col, index_col]].rolling(window, min_periods=mp).cov().unstack()
        # cov_si is a DataFrame with MultiIndex columns; pick (col, index_col)
        try:
            cov_series = cov_si[(col, index_col)]
        except KeyError:
            # fallback: compute manually at cost
            cov_series = pd.Series(index=r_df.index, dtype=float)
            for i in range(len(r_df.index)):
                j0 = i - window + 1
                if j0 < 0: 
                    continue
                block = r_df[[col, index_col]].iloc[j0:i+1].dropna()
                if len(block) >= mp:
                    cov_series.iloc[i] = block.cov().iloc[0, 1]
        # β_t = cov_t / var_i_t
        with np.errstate(divide='ignore', invalid='ignore'):
            out[col] = cov_series.values / var_i.values
    return out


# === High-level builders for tables & selections =============================

def normal_fit_table(
    r_df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    ddof: int = 1,
    annualize: bool = True,
    ann_factor: float = np.sqrt(252.0),
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Per-symbol Normal fit on the FULL sample (not rolling).
    Returns columns: symbol, n, mu, sigma, ann_sigma.

    - r_df: returns DataFrame (daily log/simple—use consistently).
    - cols: optional subset (single symbol or basket).
    - ddof: standard deviation degrees of freedom.
    - annualize: multiply σ by √252.
    - min_obs: below this, returns NaNs for μ/σ.
    """
    df = _ensure_dtindex(r_df.astype(float))
    if cols is not None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns in r_df: {sorted(missing)}")
        df = df.loc[:, list(cols)]

    out = []
    for sym, s in df.items():
        s = s.dropna()
        n = int(s.shape[0])
        if n < max(min_obs, ddof + 1):
            mu = sd = ann = np.nan
        else:
            mu = float(s.mean())
            sd = float(s.std(ddof=ddof))
            ann = float(sd * (ann_factor if annualize else 1.0))
        out.append((sym, n, mu, sd, ann))

    res = pd.DataFrame(out, columns=["symbol", "n", "mu", "sigma", "ann_sigma"])
    return res.sort_values("symbol").reset_index(drop=True)


def rolling_stat(
    df: pd.DataFrame,
    stat: str,
    window: int,
    cols: Optional[Sequence[str]] = None,
    min_frac: float = 0.80,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Rolling STAT time-series for a basket (or single symbol).
    stat ∈ {'z','skew','kurt','mean','std'}.

    - For 'z': z = (x - rolling_mean) / rolling_std
    - For 'skew','kurt': pandas' rolling implementations (kurt = excess)
    - For 'mean','std': convenience wrappers
    """
    df = _ensure_dtindex(df.astype(float))
    if cols is not None:
        keep = [c for c in cols if c in df.columns]
        if not keep:
            raise KeyError("None of the requested cols are in df.")
        df = df[keep]
    mp = _minp(window, min_frac)

    if stat == "z":
        mu = df.rolling(window, min_periods=mp).mean()
        sd = df.rolling(window, min_periods=mp).std(ddof=ddof)
        return (df - mu) / sd
    elif stat == "skew":
        return df.rolling(window, min_periods=mp).skew()
    elif stat == "kurt":
        return df.rolling(window, min_periods=mp).kurt()
    elif stat == "mean":
        mu, _ = rolling_mean_std(df, window, min_frac=min_frac, ddof=ddof)
        return mu
    elif stat == "std":
        _, sd = rolling_mean_std(df, window, min_frac=min_frac, ddof=ddof)
        return sd
    else:
        raise ValueError("stat must be one of {'z','skew','kurt','mean','std'}")


def _last_valid(x: pd.Series) -> float:
    """Return last non-NaN value or NaN if none."""
    x = pd.Series(x).dropna()
    return float(x.iloc[-1]) if not x.empty else np.nan


def rolling_snapshot_table(
    r_df: pd.DataFrame,
    windows: Sequence[int] = (5, 21, 63, 252),
    cols: Optional[Sequence[str]] = None,
    min_frac: float = 0.80,
    ddof: int = 1,
    include: Sequence[str] = ("z", "skew", "kurt"),
) -> pd.DataFrame:
    """
    Snapshot of the *latest available* rolling stats per window.

    Returns a flat table:
      symbol, [z_last_w{W}], [skew_last_w{W}], [kurt_last_w{W}], ...

    - Pick any subset via cols=... (supports single name).
    - Use include=... to choose which stats to compute.
    """
    df = _ensure_dtindex(r_df.astype(float))
    if cols is not None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns in r_df: {sorted(missing)}")
        df = df.loc[:, list(cols)]

    syms = list(df.columns)
    rows = {sym: {} for sym in syms}

    for w in windows:
        mp = _minp(w, min_frac)
        if "z" in include:
            mu = df.rolling(w, min_periods=mp).mean()
            sd = df.rolling(w, min_periods=mp).std(ddof=ddof)
            z = (df - mu) / sd
            for sym in syms:
                rows[sym][f"z_last_w{w}"] = _last_valid(z[sym])

        if "skew" in include:
            sk = df.rolling(w, min_periods=mp).skew()
            for sym in syms:
                rows[sym][f"skew_last_w{w}"] = _last_valid(sk[sym])

        if "kurt" in include:
            ku = df.rolling(w, min_periods=mp).kurt()
            for sym in syms:
                rows[sym][f"kurt_last_w{w}"] = _last_valid(ku[sym])

    out = []
    for sym in syms:
        row = {"symbol": sym}
        row.update(rows[sym])
        out.append(row)
    return pd.DataFrame(out).sort_values("symbol").reset_index(drop=True)


def build_stats_tables(
    r_df: pd.DataFrame,
    windows: Sequence[int] = (5, 21, 63, 252),
    cols: Optional[Sequence[str]] = None,
    min_frac: float = 0.80,
    ddof: int = 1,
    include: Sequence[str] = ("z", "skew", "kurt"),
    include_normal: bool = True,
    annualize_normal: bool = True,
) -> Dict[str, object]:
    """
    Orchestrator: returns a dict with
      - 'normal_full' : per-symbol μ/σ/ann_σ/n (full-sample)
      - 'snapshots'   : last z/skew/kurt per window
      - 'roll_z'      : {window -> DataFrame of rolling z}
      - 'roll_skew'   : {window -> DataFrame of rolling skew}
      - 'roll_kurt'   : {window -> DataFrame of rolling kurt}

    Use cols=... to narrow to a symbol, basket, or include the index.
    """
    out: Dict[str, object] = {}
    if include_normal:
        out["normal_full"] = normal_fit_table(
            r_df, cols=cols, ddof=ddof, annualize=annualize_normal
        )

    out["snapshots"] = rolling_snapshot_table(
        r_df, windows=windows, cols=cols, min_frac=min_frac, ddof=ddof, include=include
    )

    # Optional rolling outputs for downstream visualization
    for stat in include:
        key = f"roll_{stat}"
        out[key] = {}
        for w in windows:
            out[key][w] = rolling_stat(
                r_df, stat=stat, window=w, cols=cols, min_frac=min_frac, ddof=ddof
            )
    return out

