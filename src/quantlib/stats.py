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
from typing import Tuple, Dict, Optional

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
