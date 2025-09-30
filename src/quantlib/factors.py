# src/quantlib/factors.py
# Lightweight CAPM-style decomposition + helpers (pandas 1.2–friendly)

from typing import Optional, Sequence, Dict, List
import numpy as np
import pandas as pd

def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _minp(window: int, min_frac: float) -> int:
    return max(2, int(np.ceil(window * float(min_frac))))

def trailing_log_return(s: pd.Series, window: int, min_frac: float = 0.8) -> pd.Series:
    """
    Trailing log return over 'window' days (sum of daily log returns).
    Requires >= min_frac * window non-NaNs.
    """
    mp = _minp(window, min_frac)
    return s.rolling(window, min_periods=mp).sum()

def capm_decompose_snapshot(
    r_df: pd.DataFrame,
    index_col: str,
    window: int,
    cols: Optional[Sequence[str]] = None,
    min_frac: float = 0.8,
) -> pd.DataFrame:
    """
    Snapshot at the latest date:
      symbol, R_s (log), R_m (log), beta, sys_ret=beta*R_m, alpha=R_s-sys_ret
    - r_df: daily log-return matrix (columns = symbols incl. index_col)
    - window: trailing window length in trading days
    """
    df = _ensure_dtindex(r_df.astype(float))
    if index_col not in df.columns:
        raise KeyError(f"Index column '{index_col}' not in r_df.")
    if cols is None:
        cols = [c for c in df.columns if c != index_col]
    else:
        cols = [c for c in cols if c in df.columns and c != index_col]
        if not cols:
            raise ValueError("No valid stock columns found in r_df for 'cols'.")

    mp = _minp(window, min_frac)
    # Trailing (log) returns
    Rm = trailing_log_return(df[index_col], window, min_frac=min_frac)
    out_rows: List[Dict[str, float]] = []

    # For each stock, compute trailing return and beta in the same window
    for sym in cols:
        Rs = trailing_log_return(df[sym], window, min_frac=min_frac)
        # rolling cov/var (pairwise) → beta_t = cov(s, m)/var(m)
        pair = df[[sym, index_col]]
        cov_sm = pair.rolling(window, min_periods=mp).cov().unstack()
        try:
            cov_series = cov_sm[(sym, index_col)]
        except KeyError:
            cov_series = pd.Series(index=df.index, dtype=float)
        var_m = df[index_col].rolling(window, min_periods=mp).var(ddof=1)

        # take the latest available snapshot
        last = df.index[-1]
        r_s = float(Rs.get(last, np.nan))
        r_m = float(Rm.get(last, np.nan))
        cov_last = float(cov_series.get(last, np.nan))
        var_last = float(var_m.get(last, np.nan))

        beta = (cov_last / var_last) if (np.isfinite(var_last) and var_last > 0) else np.nan
        sys_ret = beta * r_m if np.isfinite(beta) and np.isfinite(r_m) else np.nan
        alpha = r_s - sys_ret if np.isfinite(r_s) and np.isfinite(sys_ret) else np.nan

        out_rows.append({
            "symbol": sym, "window": window,
            "R_s": r_s, "R_m": r_m, "beta": beta,
            "sys_ret": sys_ret, "alpha": alpha
        })

    res = pd.DataFrame(out_rows).sort_values("symbol").reset_index(drop=True)
    return res

def capm_decompose_table(
    r_df: pd.DataFrame,
    index_col: str,
    windows: Sequence[int] = (1, 5, 21, 63, 252),
    cols: Optional[Sequence[str]] = None,
    min_frac: float = 0.8,
) -> pd.DataFrame:
    """
    Stack snapshots for multiple windows.
    Returns columns: symbol, window, R_s, R_m, beta, sys_ret, alpha
    """
    frames = []
    for w in windows:
        frames.append(capm_decompose_snapshot(r_df, index_col=index_col, window=w, cols=cols, min_frac=min_frac))
    if not frames:
        return pd.DataFrame(columns=["symbol","window","R_s","R_m","beta","sys_ret","alpha"])
    return pd.concat(frames, axis=0).reset_index(drop=True)
