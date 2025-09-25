# src/quantlib/viz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from pathlib import Path
from typing import Iterable
from typing import Dict, Optional, Sequence, Tuple
from scipy.stats import gaussian_kde, norm, t as student_t

# --------------------------- helpers ---------------------------

def _ensure_series(x) -> pd.Series:
    s = pd.Series(x)
    s = s.astype(float)
    s = s.sort_index()
    return s

def _ensure_df(x) -> pd.DataFrame:
    df = pd.DataFrame(x).astype(float).sort_index()
    return df

def _savefig(save_path: Optional[Path], bbox_inches="tight", dpi=120) -> None:
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, bbox_inches=bbox_inches, dpi=dpi)


    


# === VaR PDF snapshot ===
def plot_var_pdf_snapshot(
    mu: float,
    sigma: float,
    alphas: Iterable[float] = (0.95, 0.99),
    dist: str = "normal",
    t_df: int = None,
    t_scale_from_sigma: bool = True,
    title: str = "VaR snapshot (latest)",
    x_limits=None,
    x_percent: bool = True,
    save_path=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    if not np.isfinite(mu) or not (np.isfinite(sigma) and sigma > 0):
        raise ValueError("mu/sigma must be finite; sigma>0")

    if x_limits is None:
        x_min, x_max = mu - 5.0 * sigma, mu + 5.0 * sigma
    else:
        x_min, x_max = x_limits
    x = np.linspace(x_min, x_max, 800)

    if dist == "normal":
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        qfun = lambda a: norm.ppf(1.0 - a, loc=mu, scale=sigma)
    elif dist == "t":
        if t_df is None or t_df <= 2:
            raise ValueError("For dist='t', provide t_df > 2")
        scale = sigma * np.sqrt((t_df - 2.0) / t_df) if t_scale_from_sigma else sigma
        pdf = student_t.pdf(x, df=t_df, loc=mu, scale=scale)
        qfun = lambda a: student_t.ppf(1.0 - a, df=t_df, loc=mu, scale=scale)
    else:
        raise ValueError("dist must be 'normal' or 't'")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, pdf, lw=2.0, label=f"{dist} PDF")

    # μ and ±1/2/3σ
    for k, ls in zip([0,1,2,3], ["-","--","--","--"]):
        if k == 0:
            ax.axvline(mu, color="#444", lw=1.2, ls=ls, alpha=0.8, label="μ")
        else:
            ax.axvline(mu + k*sigma, color="#888", lw=1.0, ls=ls, alpha=0.7)
            ax.axvline(mu - k*sigma, color="#888", lw=1.0, ls=ls, alpha=0.7)

    # VaR shading
    colors = ["#d62728", "#9467bd", "#2ca02c", "#1f77b4"]
    for i, a in enumerate(alphas):
        q_left = qfun(a)                  # threshold on return scale
        var_mag = -q_left                 # positive magnitude
        mask = x <= q_left
        ax.fill_between(x[mask], 0, pdf[mask], color=colors[i % len(colors)], alpha=0.25)
        ax.axvline(q_left, color=colors[i % len(colors)], lw=1.8,
                   label=f"VaR {int(a*100)}% ({var_mag*100:.2f}%)")

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("density")
    if x_percent:
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    try:
        from pathlib import Path
        if save_path is not None:
            p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, bbox_inches="tight", dpi=120)
    except Exception:
        pass
    plt.show()


# === VaR forecast path from sigma forecast ===
def plot_var_forecast_path(
    sigma_future: pd.Series,
    alphas: Iterable[float] = (0.95, 0.99),
    mu: float = 0.0,
    dist: str = "normal",
    t_df: int = None,
    title: str = "Forward VaR (from σ forecast)",
    save_path=None,
) -> pd.DataFrame:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import pandas as pd
    from pathlib import Path

    s = pd.Series(sigma_future).astype(float).dropna()
    if s.empty:
        raise ValueError("sigma_future is empty")

    out = pd.DataFrame(index=s.index)
    if dist == "normal":
        for a in alphas:
            out[f"VaR_{int(a*100)}"] = s.apply(lambda sd: -(mu + norm.ppf(1.0 - a, loc=mu, scale=sd)))
    elif dist == "t":
        if t_df is None or t_df <= 2:
            raise ValueError("For dist='t', provide t_df > 2")
        scale = s * np.sqrt((t_df - 2.0) / t_df)  # match daily sigma
        for a in alphas:
            out[f"VaR_{int(a*100)}"] = scale.apply(lambda sc: -(mu + student_t.ppf(1.0 - a, df=t_df, loc=mu, scale=sc)))
    else:
        raise ValueError("dist must be 'normal' or 't'")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for a in alphas:
        ax.plot(out.index, out[f"VaR_{int(a*100)}"]*100.0, lw=1.8, label=f"{int(a*100)}%")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("VaR (daily, %)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Confidence")
    try:
        if save_path is not None:
            p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, bbox_inches="tight", dpi=120)
    except Exception:
        pass
    plt.show()

    return out



# === Plot H-day VaR/ES curves ==============================================

def plot_var_es_horizon_curves(
    curves: pd.DataFrame,
    alphas: Iterable[float] = (0.95, 0.99),
    title: str = "H-day VaR / ES (from σ forecast)",
    as_percent: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot VaR_H and ES_H versus horizon H from the DataFrame emitted by
    horizon_var_es_from_forward_sigma(...).
    """
    if curves is None or curves.empty:
        raise ValueError("curves is empty")
    H = curves.index.values

    fig, ax = plt.subplots(figsize=(10, 5))
    for a, color in zip(alphas, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]):
        vcol = f"VaR_{int(a*100)}"
        ecol = f"ES_{int(a*100)}"
        if vcol in curves:
            ax.plot(H, curves[vcol], lw=2.0, color=color, label=f"VaR {int(a*100)}%")
        if ecol in curves:
            ax.plot(H, curves[ecol], lw=1.6, color=color, ls="--", label=f"ES {int(a*100)}%")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("H-day return magnitude")
    if as_percent:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        # If values are raw returns (not in 0..1), convert to % on the fly:
        ymax = np.nanmax(curves.filter(like="VaR").values)
        if ymax > 1.0:
            # plot uses raw numbers; convert tick labels only when as_percent=True and values ~ fraction
            pass
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.show()
    

# Note: We treat VaR/ES values as fractions (e.g., 0.02 = 2%) — consistent with returns

# ---------------------- 1) Correlation heatmap ----------------------

def plot_corr_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlation matrix",
    save_path: Optional[Path] = None,
    annotate_if_small: bool = False,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """
    Visualize pairwise correlations as a heatmap.
    Why this: correlations are relational — color grids reveal clusters/diversification faster than tables.

    - NaNs (insufficient overlap) shown as light grey (so missing info is explicit).
    - Diverging colormap, centered at 0.
    - Optional numeric annotations for small matrices (<= 30 x 30 recommended).
    """
    C = _ensure_df(corr_df)
    # Mask NaNs so we can color them distinctly
    A = np.ma.masked_invalid(C.values)
    fig, ax = plt.subplots(figsize=(min(12, 0.35 * C.shape[1] + 4),
                                    min(12, 0.35 * C.shape[0] + 4)))
    cmap = plt.cm.RdBu_r
    # set NaN cells to light grey; MPL 3.3: modify cmap in place
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(C.shape[1]))
    ax.set_yticks(np.arange(C.shape[0]))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(C.index, fontsize=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("corr", rotation=270, labelpad=12)

    # annotations
    if annotate_if_small and C.size <= 900:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                val = C.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")

    ax.grid(False)
    fig.tight_layout()
    _savefig(save_path)
    plt.show()

# -------- 2) PDFs / Distributions: hist + KDE + Normal / Student-t overlays --------

def plot_multi_tf_pdf(
    data: Dict[str, pd.Series],
    title: str = "PDFs across horizons",
    bins: int = 60,
    show_kde: bool = True,
    kde_bw: Optional[float] = None,   # None → Scott's rule; float → bandwidth factor
    show_normal: bool = True,
    show_student: bool = False,
    x_percent: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot empirical PDFs for multiple horizons on the same axes.

    Why this: combines empirical shape (hist) with transparent smoothing (KDE) and simple models (Normal/t)
    to compare peak, tails, and how width scales with horizon.

    - Each dict entry key is a label, value is a return series (1D, same units).
    - KDE bandwidth is explicit (avoid 'black box smoothing').
    - Normal overlay uses sample (mu, sigma). Student-t overlay is MLE-fitted (ν, loc, scale).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose consistent x-range across series
    xs_all = np.concatenate([_ensure_series(s).dropna().values for s in data.values() if len(s.dropna())])
    x_min, x_max = np.percentile(xs_all, [0.5, 99.5])  # robust range
    x_grid = np.linspace(x_min, x_max, 400)

    for label, s in data.items():
        s = _ensure_series(s).dropna()
        if s.empty:
            continue
        # histogram in density mode (area = 1)
        ax.hist(s.values, bins=bins, density=True, alpha=0.25, label=f"{label} (hist)")

        # KDE overlay
        if show_kde and len(s) > 10:
            kde = gaussian_kde(s.values, bw_method=kde_bw)
            ax.plot(x_grid, kde(x_grid), lw=1.8, label=f"{label} (kde)")

        # Normal overlay (sample moments)
        if show_normal:
            mu, sd = float(s.mean()), float(s.std(ddof=1))
            if np.isfinite(sd) and sd > 0:
                ax.plot(x_grid, norm.pdf(x_grid, loc=mu, scale=sd), lw=1.5, ls="--", label=f"{label} (normal)")

        # Student-t overlay (MLE)
        if show_student and len(s) > 20:
            try:
                nu, loc, scale = student_t.fit(s.values)  # can be slow on very long tails but ok here
                ax.plot(x_grid, student_t.pdf(x_grid, df=nu, loc=loc, scale=scale), lw=1.5, ls=":", label=f"{label} (t)")
            except Exception:
                pass  # if fit fails, just skip

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("density")
    if x_percent:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v*100:.1f}%"))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _savefig(save_path)
    plt.show()

# ------------- 3) Portfolio value (₹) & drawdown (on twin axis) --------------

def plot_value_and_drawdown(
    value_series: pd.Series,
    title: str = "Portfolio value & drawdown",
    currency: str = "₹",
    save_path: Optional[Path] = None,
) -> None:
    """
    Why this: value curve translates returns to rupees (manager's language); drawdown shows lived risk better than σ.

    - Left axis: value in INR (thousands format).
    - Right axis: drawdown % from running max.
    """
    v = _ensure_series(value_series).dropna()
    if v.empty:
        return
    roll_max = v.cummax()
    dd = v / roll_max - 1.0

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(v.index, v.values, lw=1.8, color="#1f77b4", label="Value")
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel(f"value ({currency})")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}"))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(dd.index, dd.values, lw=1.4, color="#d62728", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("drawdown")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    # build a combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    fig.tight_layout()
    _savefig(save_path)
    plt.show()

# ------------------- 4) VaR band with exceptions (backtest view) -------------------

def plot_var_exceptions(
    r: pd.Series,
    var: pd.Series,
    title: str = "VaR backtest",
    alpha_label: str = "95%",
    save_path: Optional[Path] = None,
    footnote: Optional[str] = None,
) -> None:
    """
    Why this: visually shows when realized losses pierce the model envelope (where, how often, any clustering).

    - r: realized daily returns (level plot, light)
    - var: VaR magnitudes (positive); threshold is -var on return scale
    - red markers on breaches
    """
    r = _ensure_series(r)
    var = _ensure_series(var)
    # align
    idx = r.index.intersection(var.index)
    r = r.reindex(idx)
    var = var.reindex(idx)

    thr = -var
    breaches = r < thr

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(r.index, r.values, lw=1.0, color="#666666", alpha=0.7, label="returns")
    ax.plot(thr.index, thr.values, lw=1.6, color="#2ca02c", label=f"-VaR ({alpha_label})")
    ax.scatter(r.index[breaches], r.values[breaches], color="#d62728", s=12, zorder=3, label="exceptions")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v*100:.1f}%"))
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    if footnote:
        ax.text(0.99, -0.12, footnote, transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444444")

    fig.tight_layout()
    _savefig(save_path)
    plt.show()

# --------------------- 5) Volatility comparison / panels ---------------------

def plot_vol_series(
    series_dict: Dict[str, pd.Series],
    title: str = "Volatility comparison",
    annualize: bool = False,
    ann_factor: float = np.sqrt(252.0),
    save_path: Optional[Path] = None,
) -> None:
    """
    Single-axis overlay for volatility estimators.
    Why this: quick signal-speed comparison (realized vs EWMA vs forward finite-EWMA).

    series_dict: label -> sigma series (daily units). If annualize, multiply by ann_factor.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, s in series_dict.items():
        s = _ensure_series(s).dropna()
        if s.empty: 
            continue
        y = s.values * (ann_factor if annualize else 1.0)
        ax.plot(s.index, y, lw=1.6, label=label)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("σ (annualized)" if annualize else "σ (daily)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v*100:.1f}%"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(save_path)
    plt.show()

def plot_vol_panels(
    panels: Dict[str, Dict[str, pd.Series]],
    title: str = "Volatility panels",
    annualize: bool = True,
    ann_factor: float = np.sqrt(252.0),
    ncols: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """
    Grid of subplots by horizon (e.g., {"5d": {...}, "21d": {...}}).
    Why this: separates horizons visually but keeps estimator comparison local to each panel.

    panels: {panel_title -> {line_label -> sigma_series}}
    """
    n = len(panels)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows), squeeze=False)

    for ax, (ptitle, series_dict) in zip(axes.ravel(), panels.items()):
        for label, s in series_dict.items():
            s = _ensure_series(s).dropna()
            if s.empty:
                continue
            y = s.values * (ann_factor if annualize else 1.0)
            ax.plot(s.index, y, lw=1.4, label=label)
        ax.set_title(ptitle, fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v*100:.1f}%"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    # hide any unused axes
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    _savefig(save_path)
    plt.show()

# ----------------------- 6) Weights: pie & ranked bars -----------------------

def plot_weights_pie_and_bars(
    weights: pd.Series,
    title: str = "Portfolio weights",
    top_n: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Why this: pies give quick overview; ranked bars reveal concentration & long tail.
    - weights: Series index = symbols, values = weights (0..1). Will be renormalized if sum != 1.
    - top_n: show top-N in bars; remaining mass is aggregated as 'OTHER'.
    """
    w = _ensure_series(weights).dropna()
    s = float(w.sum())
    if s <= 0:
        return
    w = w / s

    # Pie (small)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 2.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Pie: show at most 12 slices for readability
    w_desc = w.sort_values(ascending=False)
    if len(w_desc) > 12:
        w_pie = w_desc.iloc[:11]
        w_pie["OTHER"] = w_desc.iloc[11:].sum()
    else:
        w_pie = w_desc

    ax1.pie(w_pie.values, labels=w_pie.index, autopct=lambda p: f"{p:.1f}%", startangle=90, textprops={"fontsize": 8})
    ax1.set_title("Weights (pie)")

    # Bars: top_n (optional)
    if top_n is not None and top_n < len(w_desc):
        w_bar = w_desc.iloc[:top_n]
        w_bar["OTHER"] = w_desc.iloc[top_n:].sum()
    else:
        w_bar = w_desc

    ax2.bar(w_bar.index, w_bar.values)
    ax2.set_title("Weights (ranked)")
    ax2.set_ylabel("weight")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=75, labelsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _savefig(save_path)
    plt.show()

    
# --- Forward-vol overlay (historical vs future) GARCH (1,1)---------------------------
    
def plot_vol_forecast_overlay(
    hist_sig: Dict[str, pd.Series],      # {"Realized 21d": series, ...}
    fut_sig: Dict[str, pd.Series],       # {"EWMA 21d λ=0.94": series_future, "GARCH(1,1)": series_future, ...}
    title: str = "Volatility — realized vs forward paths",
    annualize: bool = True,              # daily σ → annualize by √252
    y_percent: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot historical realized sigma (solid) and forward sigma forecasts (dashed)
    on the same axis. Future region is lightly shaded; a vertical line marks 'today'.
    """
    if not hist_sig:
        raise ValueError("hist_sig is empty")
    if not fut_sig:
        raise ValueError("fut_sig is empty")

    # Clean & align
    hist_clean = {k: pd.Series(v).astype(float).dropna() for k, v in hist_sig.items()}
    fut_clean  = {k: pd.Series(v).astype(float).dropna() for k, v in fut_sig.items()}

    # Last observed date = max end among historical series
    last_date = max(s.index.max() for s in hist_clean.values())
    fut_max   = max(s.index.max() for s in fut_clean.values())

    # Annualize if requested
    scale = np.sqrt(252.0) if annualize else 1.0
    hist_scaled = {k: s * scale for k, s in hist_clean.items()}
    fut_scaled  = {k: s * scale for k, s in fut_clean.items()}

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot historical (solid)
    for label, s in hist_scaled.items():
        s_plot = s[s.index <= last_date]
        ax.plot(s_plot.index, s_plot.values, lw=1.8, label=label)

    # Future shading and split line
    ax.axvspan(last_date, fut_max, color="#f0f0f0", alpha=0.5, zorder=0)
    ax.axvline(last_date, color="#666", lw=1.0, ls=":", label="today")

    # Plot futures (dashed)
    for label, s in fut_scaled.items():
        ax.plot(s.index, s.values, lw=2.0, ls="--", label=label)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("σ (annualized)" if annualize else "σ (daily)")
    if y_percent:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    try:
        if save_path is not None:
            p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=120, bbox_inches="tight")
    except Exception:
        pass

    plt.show()
    