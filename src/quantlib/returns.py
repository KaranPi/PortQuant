# import numpy as np — vectorized math (log, isnan masks).

#import pandas as pd — DataFrame ops, rolling windows.

#from typing import Iterable, Dict, List, Optional — type hints only (readability).

#No heavy deps; 100% compatible with your environment.

# ----------------------------------------------------------------


# src/quantlib/returns.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ----------------------------- Basics & checks -----------------------------

def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DateTimeIndex, sorted & unique (keeps first if duplicates).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

# ----------------------------- Daily returns -------------------------------

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Gap-aware daily log returns for each column:
        r_t = ln(P_t) - ln(P_{t-1})
    Returns NaN where either current or previous price is NaN (pre-IPO, holidays).
    """
    prices = _ensure_dtindex(prices.astype(float))
    lnp = np.log(prices)
    r = lnp.diff()                 # same as ln(P_t / P_{t-1})
    # Ensure we don't create spurious values across gaps:
    valid = prices.notna() & prices.shift(1).notna()
    return r.where(valid)

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    (Optional) Gap-aware simple returns:
        r_t = (P_t / P_{t-1}) - 1
    """
    prices = _ensure_dtindex(prices.astype(float))
    r = prices.pct_change()
    valid = prices.notna() & prices.shift(1).notna()
    return r.where(valid)

# ----------------------- Trailing multi-horizon returns --------------------

def trailing_log_return_from_daily(lrets: pd.DataFrame, window: int,
                                   min_frac: float = 1.0) -> pd.DataFrame:
    """
    Trailing log return over 'window' days:
        R_{t,win} = sum_{i=0..win-1} r_{t-i}
    min_frac=1.0 → require a full window of valid daily lrets.
    If you want a more permissive window (e.g., 80% valid), pass min_frac=0.8.
    """
    lrets = _ensure_dtindex(lrets.astype(float))
    minp = max(1, int(np.ceil(window * float(min_frac))))
    # rolling sum of log-returns → trailing log return over 'window'
    R = lrets.rolling(window, min_periods=minp).sum()
    # Keep NaN where the *price* window was not fully observed (strict mode default)
    return R

def trailing_log_returns_pack(lrets: pd.DataFrame,
                              windows: List[int],
                              min_frac: float = 1.0) -> Dict[int, pd.DataFrame]:
    """
    Convenience: compute trailing log returns for several windows at once.
    Returns a dict: {window -> DataFrame}
    """
    out = {}
    for w in windows:
        out[w] = trailing_log_return_from_daily(lrets, w, min_frac=min_frac)
    return out
