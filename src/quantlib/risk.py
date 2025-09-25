# import numpy as np — numerics (sqrt, dot, clipping).
# import pandas as pd — rolling windows, joins, quantiles.
# from scipy.stats import chi2, norm — Kupiec p-values (χ²), Normal quantiles.
# (We’ll add Student-t in a later pass; keeping this module lean.)
# All are already in env (scipy=1.7.0).



# Everything is NaN-safe and pandas-1.2-compatible.
# VaR/ES sign convention: we return positive loss magnitudes. Thresholding uses r < -VaR.
# We compute Normal-EWMA VaR (μ=0 by default; you can swap to rolling mean if desired).
# Student-t VaR/ES can be added next (we already discussed it; keeping this module focused for now).




# -----------------------------



# src/quantlib/risk.py


import numpy as np
import pandas as pd
from typing import Iterable, Optional, Dict, Tuple
from scipy.stats import chi2,norm, t as student_t
from scipy.optimize import minimize

# === Horizon VaR/ES from forward sigma path ================================

def horizon_var_es_from_forward_sigma(
    sigma_future: pd.Series,
    alphas: Iterable[float] = (0.95, 0.99),
    dist: str = "normal",           # "normal" or "t"
    t_df: Optional[int] = None,     # required if dist="t"
    mu: float = 0.0,                # per-day drift; default 0
    H_max: Optional[int] = None,    # default: use full length of sigma_future
) -> pd.DataFrame:
    """
    Convert a forward daily sigma path into H-day VaR/ES curves for H=1..H_max.

    Parameters
    ----------
    sigma_future : pd.Series
        Future per-day volatility (std) forecast indexed by future business dates.
    alphas : iterable of float
        Confidence levels (e.g., 0.95, 0.99). VaR is the (1-alpha)% left-tail magnitude.
    dist : {"normal","t"}
        Parametric family for VaR/ES conversion.
    t_df : int, optional
        Degrees of freedom for Student-t (must be > 2).
    mu : float
        Per-day drift for H-day horizon (used as H*mu in aggregation). Usually 0.
    H_max : int, optional
        Maximum horizon; default uses full length of sigma_future.

    Returns
    -------
    pd.DataFrame
        Index: H = 1..H_max.
        Columns: 'sigma_H' plus 'VaR_{int(alpha*100)}', 'ES_{int(alpha*100)}' for each alpha.
        Values are positive magnitudes on H-day return scale (not annualized).
    """
    s = pd.Series(sigma_future).astype(float).dropna()
    if s.empty:
        raise ValueError("sigma_future is empty")
    n = len(s)
    H_max = int(H_max) if H_max is not None else n
    H_max = max(1, min(H_max, n))

    # Aggregate horizon sigma: sqrt of cumulative sum of daily variances
    sig2_cum = np.cumsum(np.square(s.values))
    sigma_H = np.sqrt(sig2_cum[:H_max])  # length H_max

    # Prepare output
    idx = pd.Index(np.arange(1, H_max + 1), name="H")
    out = pd.DataFrame({"sigma_H": sigma_H}, index=idx)

    # VaR/ES per alpha
    for a in alphas:
        colV = f"VaR_{int(a*100)}"
        colE = f"ES_{int(a*100)}"

        if dist == "normal":
            z = norm.ppf(a)                    # e.g., 1.6449 for 95%
            phi = norm.pdf(z)
            # VaR_H = z * sigma_H ; ES_H = (phi/(1-alpha)) * sigma_H
            out[colV] = z * out["sigma_H"]
            out[colE] = (phi / (1.0 - a)) * out["sigma_H"]

        elif dist == "t":
            if t_df is None or t_df <= 2:
                raise ValueError("For dist='t', provide t_df > 2")
            # Match std:  sigma_H^2 = scale_H^2 * df/(df-2)  => scale_H = sigma_H * sqrt((df-2)/df)
            scale_H = out["sigma_H"] * np.sqrt((t_df - 2.0) / t_df)
            # Left-tail quantile magnitude (use right-tail symmetry): q_mag = t.ppf(a, df)
            q_mag = student_t.ppf(a, df=t_df)
            # For ES (left-tail) of Student-t(0, scale, df):
            # ES = scale * [ (df + q^2) / ((df-1)*(1-a)) ] * f_t(q)
            f_q = student_t.pdf(q_mag, df=t_df)            # pdf at +q (symmetry)
            factor = ((t_df + q_mag**2) / ((t_df - 1.0) * (1.0 - a))) * f_q
            out[colV] = q_mag * out["sigma_H"] / (np.sqrt(t_df / (t_df - 2.0)))  # = q * scale_H
            out[colE] = factor * scale_H

        else:
            raise ValueError("dist must be 'normal' or 't'")

        # incorporate H*mu if you want drift in VaR level; magnitude typically reported without drift
        # We keep VaR/ES as magnitudes from zero (consistent with earlier plots).

    return out



# ============================== Volatility ==============================

def realized_vol(x: pd.Series, window: int, min_frac: float = 0.8, ddof: int = 1) -> pd.Series:
    """
    Rolling realized volatility (std) of a return series (daily units).
    Keeps NaN if fewer than min_frac*window valid points.
    """
    x = pd.Series(x).astype(float)
    mp = max(2, int(np.ceil(window * float(min_frac))))
    return x.rolling(window, min_periods=mp).std(ddof=ddof)

def ewma_var_recursive(x: pd.Series, lam: float = 0.94, backcast_n: int = 60) -> pd.Series:
    """
    EWMA variance (daily) via the standard recursion:
        var_t = λ * var_{t-1} + (1-λ) * r_t^2
    Seed (backcast) is mean(r^2) over first backcast_n valid points.
    """
    x = pd.Series(x).astype(float)
    r2 = x * x
    out = pd.Series(index=x.index, dtype=float)
    fv = r2.first_valid_index()
    if fv is None:
        return out
    seed = float(r2.loc[fv:].dropna().iloc[:max(1, backcast_n)].mean())
    prev = seed
    for t in x.index:
        rt2 = r2.loc[t]
        curr = prev if np.isnan(rt2) else lam * prev + (1.0 - lam) * rt2
        out.loc[t] = curr
        prev = curr
    return out

def ewma_weights(lam: float, n: int) -> np.ndarray:
    """
    Finite EWMA weights α(i) for i=1..n (i=1 is most recent), normalized to sum=1:
        α(i) = [(1-λ) * λ^(i-1)] / [1 - λ^n]
    """
    i = np.arange(0, n, dtype=float)  # 0..n-1
    numer = (1.0 - lam) * np.power(lam, i)
    denom = 1.0 - np.power(lam, n)
    w = numer / denom if denom > 0 else numer
    # reverse to align α(1) with most recent when we multiply r_{t}, r_{t-1}, ...
    return w

def finite_window_ewma_sigma(x: pd.Series, window: int, lam: float) -> pd.Series:
    """
    Finite-window EWMA *sigma* (not variance) computed with normalized weights over *window* most recent returns at each t.
    For each date t with ≥window valid returns, σ_t = sqrt( sum_i α(i) * r_{t+1-i}^2 ).
    """
    x = pd.Series(x).astype(float)
    w = ewma_weights(lam, window)  # α(1..n) sums to 1
    out = pd.Series(index=x.index, dtype=float)
    vals = x.values
    n = len(vals)
    for idx in range(n):
        j0 = idx - window + 1
        if j0 < 0:
            continue
        block = vals[j0:idx+1]
        if np.isnan(block).sum() == 0:
            # align: α(1) with block[-1] (most recent)
            s2 = np.dot(w, block[::-1] ** 2)
            out.iloc[idx] = np.sqrt(s2)
    return out

# ============================== VaR / ES ================================

def hist_var_es(
    r: pd.Series,
    window: int = 252,
    alpha: float = 0.95,
    min_frac: float = 0.8
) -> pd.DataFrame:
    """
    Historical VaR/ES (one-sided, left tail).
    Returns positive-loss VaR, ES (i.e., VaR = -quantile at (1-alpha), ES = -mean of tail).
    """
    r = pd.Series(r).astype(float)
    mp = max(10, int(np.ceil(window * float(min_frac))))
    p = 1.0 - alpha  # tail probability (e.g., 0.05 for 95%)
    out = pd.DataFrame(index=r.index, columns=["VaR", "ES"], dtype=float)

    roll = r.rolling(window, min_periods=mp)

    # VaR as positive magnitude
    q = roll.quantile(p)  # left-tail quantile (negative)
    out["VaR"] = -q

    # ES: mean of returns <= q
    # implement via custom loop for pandas 1.2 compatibility
    for i in range(len(r.index)):
        j0 = i - window + 1
        if j0 < 0:
            continue
        block = r.iloc[j0:i+1].dropna()
        if len(block) >= mp:
            qi = q.iloc[i]
            tail = block[block <= qi]
            out.iloc[i, out.columns.get_loc("ES")] = -float(tail.mean()) if len(tail) else np.nan

    return out

def normal_var_es_from_sigma(
    mu: pd.Series,
    sigma: pd.Series,
    alpha: float = 0.95
) -> pd.DataFrame:
    """
    Parametric Normal VaR/ES given per-date μ_t and σ_t (daily).
    VaR, ES returned as positive-loss magnitudes.
    """
    mu = pd.Series(mu).astype(float)
    sigma = pd.Series(sigma).astype(float)
    z = norm.ppf(1.0 - alpha)  # e.g., 0.05-quantile (negative)
    # VaR magnitude:
    var = -(mu + sigma * z)

    # ES magnitude: for Normal, ES = σ * φ(z) / (1-α) - μ
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    es = sigma * (phi / (1.0 - alpha)) - mu

    return pd.DataFrame({"VaR": var, "ES": es}, index=mu.index)

def normal_ewma_var_es(
    r: pd.Series,
    lam: float = 0.94,
    alpha: float = 0.95,
    backcast_n: int = 60
) -> pd.DataFrame:
    """
    Parametric Normal VaR/ES using EWMA σ_t and μ_t=0 (or small rolling mean if you prefer).
    Returns daily VaR/ES magnitudes over time.
    """
    r = pd.Series(r).astype(float)
    var = ewma_var_recursive(r, lam=lam, backcast_n=backcast_n)
    sigma = np.sqrt(var)
    mu = pd.Series(0.0, index=r.index)  # set to 0; change to rolling mean if desired
    return normal_var_es_from_sigma(mu, sigma, alpha=alpha)

# ============================== Backtests ===============================

def var_exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    """
    Boolean series: True when realized return < -VaR (loss exceeds VaR).
    (VaR is positive magnitude; threshold is -VaR on return scale.)
    """
    r = pd.Series(r).astype(float)
    var = pd.Series(var).astype(float)
    return r < (-var)

def kupiec_pvalue(exceptions: pd.Series, alpha: float) -> Tuple[float, Dict[str, float]]:
    """
    Kupiec Unconditional Coverage test (LR_uc).
    H0: exception rate = p = 1-α. Returns (pvalue, stats dict).
    """
    x = exceptions.dropna().astype(bool).values
    T = x.size
    N = x.sum()
    if T == 0:
        return np.nan, {"T": 0, "N": 0, "LR_uc": np.nan}
    p = 1.0 - alpha
    phat = N / float(T) if T > 0 else 0.0
    # Guard against 0/1 edge cases
    eps = 1e-12
    L0 = (1 - p) ** (T - N) * (p ** N)
    L1 = (1 - phat + eps) ** (T - N) * ((phat + eps) ** N)
    LR_uc = -2.0 * np.log(L0 / L1) if L0 > 0 else np.inf
    pval = 1.0 - chi2.cdf(LR_uc, df=1)
    return float(pval), {"T": int(T), "N": int(N), "LR_uc": float(LR_uc), "phat": float(phat), "p": float(p)}

def christoffersen_pvalue(exceptions: pd.Series) -> Tuple[float, Dict[str, float]]:
    """
    Christoffersen Independence test (LR_ind).
    H0: exceptions are independent over time (no clustering).
    """
    x = exceptions.dropna().astype(bool).values
    if x.size < 2:
        return np.nan, {"LR_ind": np.nan}
    # transitions
    N00 = N01 = N10 = N11 = 0
    for i in range(1, x.size):
        a, b = x[i-1], x[i]
        if not a and not b: N00 += 1
        if not a and b:     N01 += 1
        if a and not b:     N10 += 1
        if a and b:         N11 += 1
    # probabilities
    def _safe_ratio(num, den): return (num / den) if den > 0 else 0.0
    pi0 = _safe_ratio(N01, N00 + N01)
    pi1 = _safe_ratio(N11, N10 + N11)
    pi  = _safe_ratio(N01 + N11, N00 + N01 + N10 + N11)
    # likelihoods
    eps = 1e-12
    L0 = ((1 - pi + eps) ** (N00 + N10)) * ((pi + eps) ** (N01 + N11))
    L1 = ((1 - pi0 + eps) ** N00) * ((pi0 + eps) ** N01) * ((1 - pi1 + eps) ** N10) * ((pi1 + eps) ** N11)
    LR_ind = -2.0 * np.log(L0 / L1) if L0 > 0 else np.inf
    pval = 1.0 - chi2.cdf(LR_ind, df=1)
    return float(pval), {"LR_ind": float(LR_ind), "N00":N00,"N01":N01,"N10":N10,"N11":N11,
                         "pi0":float(pi0),"pi1":float(pi1),"pi":float(pi)}

def traffic_light(p_uc: float, p_ind: float) -> str:
    """
    Simple Basel-style traffic light:
      - 'GREEN' if both p-values >= 0.05
      - 'YELLOW' if any in [0.01, 0.05)
      - 'RED' if any < 0.01
    """
    if np.isnan(p_uc) or np.isnan(p_ind):
        return "NA"
    if p_uc < 0.01 or p_ind < 0.01:
        return "RED"
    if p_uc < 0.05 or p_ind < 0.05:
        return "YELLOW"
    return "GREEN"


# ---------------------------- GARCH(1,1) (Normal) ----------------------------

def _garch11_from_raw(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Map unconstrained params x -> (omega>0, alpha>=0, beta>=0, alpha+beta<1).
    Uses softplus / sigmoid to enforce constraints smoothly.
    """
    # omega = exp(x0) * scale to avoid underflow
    omega = np.exp(x[0])  # >0
    a = 1.0 / (1.0 + np.exp(-x[1]))  # in (0,1)
    btil = 1.0 / (1.0 + np.exp(-x[2]))  # in (0,1)
    beta = btil * max(1e-6, 1.0 - a - 1e-6)  # ensure alpha+beta<1
    alpha = a
    return float(omega), float(alpha), float(beta)

def _garch11_nll(x: np.ndarray, r: np.ndarray) -> float:
    """
    Negative log-likelihood for GARCH(1,1) under Normal innovations.
    r assumed demeaned (or near zero mean).
    h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    """
    omega, alpha, beta = _garch11_from_raw(x)
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        return 1e6  # reject invalid region

    n = r.size
    if n < 10:
        return 1e6

    # initialize variance at unconditional variance
    h = np.empty(n)
    h0 = np.var(r, ddof=1) if np.isfinite(np.var(r, ddof=1)) and np.var(r, ddof=1) > 0 else 1e-6
    h[0] = omega / max(1e-6, (1.0 - alpha - beta)) if (alpha + beta) < 0.9999 else h0
    for t in range(1, n):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        if not np.isfinite(h[t]) or h[t] <= 0:
            return 1e6

    # Gaussian NLL: 0.5 * Σ ( log(2π) + log(h_t) + r_t^2 / h_t )
    nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + (r**2) / h)
    return float(nll)

def fit_garch11_normal(r: pd.Series) -> Dict[str, float]:
    """
    Fit GARCH(1,1) with Normal innovations. Returns dict with params & conditional variance series.
    """
    r = pd.Series(r).astype(float).dropna()
    r = r - r.mean()  # demean (simple)

    # initialize raw params: omega ~ var*(1 - 0.05 - 0.90), alpha~0.05, beta~0.90
    var0 = np.var(r.values, ddof=1)
    alpha0, beta0 = 0.05, 0.90
    omega0 = max(1e-8, var0 * max(1e-3, 1.0 - alpha0 - beta0))
    x0 = np.array([np.log(omega0), 0.0, 0.0], dtype=float)  # raw space

    res = minimize(_garch11_nll, x0, args=(r.values,), method="L-BFGS-B")
    omega, alpha, beta = _garch11_from_raw(res.x)

    # build conditional variance series with final params
    n = r.size
    h = np.empty(n)
    h[0] = omega / max(1e-6, (1.0 - alpha - beta)) if (alpha + beta) < 0.9999 else var0
    for t in range(1, n):
        h[t] = omega + alpha * r.iloc[t-1]**2 + beta * h[t-1]

    out = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "llf": -_garch11_nll(res.x, r.values),
        "converged": bool(res.success),
        "nobs": int(n),
        "sigma": pd.Series(np.sqrt(h), index=r.index, name="sigma_garch11"),
    }
    out["alpha_plus_beta"] = out["alpha"] + out["beta"]
    return out

def garch11_forecast_path(params: Dict[str, float], last_sigma: float, horizon: int, last_date: pd.Timestamp) -> pd.Series:
    """
    Forecast per-day sigma for the next 'horizon' business days using
    h_{t+1} = omega + (alpha + beta) * h_t (since E[r_t^2] = h_t under Normal).
    """
    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta  = float(params["beta"])
    s2 = float(last_sigma)**2

    out = np.empty(horizon)
    for h in range(horizon):
        s2 = omega + (alpha + beta) * s2
        out[h] = np.sqrt(s2)

    fut_idx = pd.date_range(last_date + pd.offsets.BDay(1), periods=horizon, freq="B")
    return pd.Series(out, index=fut_idx, name="sigma_garch11_forecast")

def vol_forecast_path(
    r: pd.Series,
    model: str = "ewma",     # "ewma" or "garch11" or "arch"
    horizon: int = 60,
    lam: float = 0.94,
    ewma_window: int = 21,
) -> pd.Series:
    """
    Unified interface: returns per-day sigma path for the next 'horizon' business days.
    - "ewma": flat forecast at last EWMA sigma (same idea you used before for forward VaR paths).
    - "garch11": fit GARCH(1,1) and use recursive variance forecast.
    - "arch": same as garch with beta=0 (by fitting and then forcing beta=0).
    """
    r = pd.Series(r).astype(float).dropna()
    last_date = r.index[-1]

    if model.lower() == "ewma":
        # use your existing recursion to get full sigma series; then hold last value flat into the future
        # reuse ewma_var_recursive already in risk.py; if not, fallback to a finite-window approx
        try:
            sig = ewma_var_recursive(r, lam=lam).pow(0.5)
        except NameError:
            # finite-window fallback using your earlier helper idea:
            w = (1 - lam) * (lam ** np.arange(ewma_window-1, -1, -1))
            w = w / w.sum()
            sig = r.rolling(ewma_window).apply(lambda x: np.sqrt(np.dot(w, x[::-1]**2)), raw=True)
        last_sigma = float(sig.dropna().iloc[-1])
        return pd.Series(last_sigma, index=pd.date_range(last_date + pd.offsets.BDay(1), periods=horizon, freq="B"),
                         name=f"sigma_{model}_forecast")

    elif model.lower() in ("garch11", "arch"):
        fit = fit_garch11_normal(r)
        if not fit["converged"]:
            # fall back to EWMA if optimizer fails
            return vol_forecast_path(r, model="ewma", horizon=horizon, lam=lam, ewma_window=ewma_window)
        params = {"omega": fit["omega"], "alpha": fit["alpha"], "beta": fit["beta"]}
        if model.lower() == "arch":
            params["beta"] = 0.0  # ARCH is GARCH with beta=0
        last_sigma = float(fit["sigma"].iloc[-1])
        return garch11_forecast_path(params, last_sigma, horizon, last_date)

    else:
        raise ValueError("model must be 'ewma', 'garch11', or 'arch'")

        
def garch11_forward_like_ewma(
    r: pd.Series,
    window: int = 252,
    horizon: int = 60,
) -> pd.Series:
    """
    'EWMA-style' forward for GARCH(1,1):
    For k=0..horizon-1, fit GARCH(1,1) on the last `window` returns ending at t-k,
    then take the *1-step-ahead* forecast for date t+1+k.

    Returns a future-dated per-day sigma series (index = business days).
    """
    r = pd.Series(r).astype(float).dropna()
    n = len(r)
    if n < window + 5:
        raise ValueError("Not enough data for the chosen window.")

    vals = []
    for k in range(horizon):
        end = n - k
        start = end - window
        if start < 1:
            break
        r_train = r.iloc[start:end]
        fit = fit_garch11_normal(r_train)
        if not fit["converged"]:
            vals.append(np.nan)
            continue
        # 1-step-ahead conditional variance:
        omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
        last_sigma = float(fit["sigma"].iloc[-1])
        rt = float(r_train.iloc[-1])
        s2_next = omega + alpha * (rt ** 2) + beta * (last_sigma ** 2)
        vals.append(np.sqrt(s2_next))

    fut_idx = pd.date_range(r.index[-1] + pd.offsets.BDay(1), periods=len(vals), freq="B")
    # mirror your finite-EWMA helper’s ordering
    return pd.Series(vals[::-1], index=fut_idx, name=f"sigma_garch11_fwd_like_ewma_w{window}")
