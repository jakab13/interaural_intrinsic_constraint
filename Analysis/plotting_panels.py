# Analysis/plotting_panels.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("MacOSX")  # for PyCharm on macOS
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

DEFAULT_CSV = Path("Dataframes/psychometrics_params_selected_model_pred.csv")
OUT_DIR = Path("Plots")
X_MIN, X_MAX = -5.0, 25.0
ARROW_WEAK2STRONG = "#7F7F7F"  # dark grey
ARROW_STRONG2WEAK = "#C7C7C7"  # light grey
# Panel C greys (dots are 10/255 darker than bars)
BAR_WEAK2STRONG = ARROW_WEAK2STRONG  # "#7F7F7F"
BAR_STRONG2WEAK = ARROW_STRONG2WEAK  # "#C7C7C7"
DOT_WEAK2STRONG = "#6b6b6b"          # 127-10 = 117
DOT_STRONG2WEAK = "#b3b3b3"          # 199-10 = 189


# Use saved psignifit results for plotting
try:
    import psignifit as ps
except Exception:
    ps = None  # we'll raise if plotting is attempted without it

def _load_result(path: str):
    if ps is None:
        raise ImportError("psignifit is not available; please import your psignifit version.")
    return ps.Result.load_json(str(path))

def _curve_on_grid_from_result(result, x_grid: np.ndarray) -> np.ndarray:
    """
    Evaluate the full psychometric (γ + (1-γ-λ)*sigmoid) on a *given* grid.
    """
    params = result.get_parameters_estimate(estimate_type="MAP")
    if params.get("gamma", None) is None:  # equal-asymptote fallback
        params["gamma"] = params["lambda"]
    sig = result.configuration.make_sigmoid()
    y = sig(x_grid, params["threshold"], params["width"])
    return (1 - params["gamma"] - params["lambda"]) * y + params["gamma"]

def _plot_curve_from_result_path(
    ax, res_path: str, line_color, line_width=1.0, line_style="-", extrapolate_stimulus=0.15, **line_kws
):
    """
    Your snippet: draw curve from saved result file + a vertical at the threshold (to 0.5).
    Returns (pse, x84) where x84 is the absolute 84% crossing.
    """
    result = _load_result(res_path)
    params = result.get_parameters_estimate(estimate_type="MAP")
    data = np.asarray(result.data)
    config = result.configuration

    if params.get("gamma", None) is None:
        params["gamma"] = params["lambda"]
    if data.size == 0:
        return None, None

    x_data = data[:, 0].astype(float)

    # build extended x and compute y (as in your snippet)
    sigmoid = config.make_sigmoid()
    x = np.linspace(X_MIN, X_MAX, num=1000)
    x_low  = np.linspace(x[0]  - extrapolate_stimulus * (x[-1] - x[0]), x[0],  num=100)
    x_high = np.linspace(x[-1],  x[-1] + extrapolate_stimulus * (x[-1] - x[0]), num=100)
    y = sigmoid(np.r_[x_low, x, x_high], params["threshold"], params["width"])
    y = (1 - params["gamma"] - params["lambda"]) * y + params["gamma"]

    # draw only the core segment
    ax.plot(x, y[len(x_low):-len(x_high)], c=line_color, lw=line_width, ls=line_style, clip_on=False, **line_kws)

    # vertical at PSE (to 0.5 as per your figure style)
    pse = float(params["threshold"])
    # ax.vlines(pse, 0.0, 0.5, color=line_color, lw=max(1.2, line_width))

    # absolute 84% crossing from result
    try:
        x84 = float(result.threshold(0.84, return_ci=False, unscaled=False))
    except Exception:
        x84 = None

    return pse, x84


# ---------- IO ----------
def load_params(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params file not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    # make sure numerics are numeric
    for c in ("standard_center_frequency","standard_angle_abs","pse","width","gamma","lambda","eta","jnd","pse_delta"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- cosmetics ----------
def trialtype_colors() -> Dict[str, tuple]:
    cmap = matplotlib.cm.get_cmap("tab20")
    return {
        "ITD-->ITD":   cmap(2),
        "ILD-->ITD":   cmap(3),
        "BOTH-->ITD":  cmap(3),
        "ILD-->ILD":   cmap(0),
        "ITD-->ILD":   cmap(1),
        "BOTH-->ILD":  cmap(1),
        "BOTH-->BOTH": cmap(4),
        "ITD-->BOTH":  cmap(5),
        "ILD-->BOTH":  cmap(5),
    }

def short_label(trial_type: str) -> str:
    """Map cues to single-letter labels: T=ITD, L=ILD, C=BOTH(Combined)."""
    m = {"ITD": "T", "ILD": "L", "BOTH": "C", "Both": "C", "combined": "C", "Combined": "C"}
    a, b = [s.strip() for s in trial_type.split("-->")]
    return f"{m.get(a, a[0])}-{m.get(b, b[0])}"

def _safe(s: object) -> str:
    return str(s).replace(":", "-").replace("/", "-").replace(" ", "_")

# ---------- psychometric curve ----------
# --- Normal CDF helper (no numpy.erf) ---
try:
    # vectorized erf if SciPy is present
    from scipy.special import erf as _erf

    def _phi(z):
        z = np.asarray(z, dtype=float)
        return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
except Exception:
    # fallback: vectorize Python's math.erf
    from math import erf as _erf_scalar

    def _phi(z):
        z = np.asarray(z, dtype=float)
        return 0.5 * (1.0 + np.vectorize(_erf_scalar)(z / np.sqrt(2.0)))

def psychometric_y(x: np.ndarray, threshold: float, width: float, gamma: float, lam: float, eta: float = 0.0) -> np.ndarray:
    """psignifit 'norm' sigmoid with equal asymptote: y = gamma + (1-gamma-lambda)*Phi((x-thresh)/width)"""
    base = _phi((x - float(threshold)) / float(width))
    return float(gamma) + (1.0 - float(gamma) - float(lam)) * base

# ---------- helpers ----------
def _is_within_cue(df: pd.DataFrame) -> pd.Series:
    return df["standard_cue"].astype(str) == df["comparison_cue"].astype(str)

def _select_experiment(df: pd.DataFrame, dataset: str, frequency: Optional[float]) -> pd.DataFrame:
    out = df[df["dataset"] == dataset].copy()
    if dataset == "combined_cue":
        if frequency is None:
            raise ValueError("For combined_cue please pass frequency=500.0 or 1300.0")
        out = out[np.isclose(pd.to_numeric(out["standard_center_frequency"], errors="coerce"), float(frequency), equal_nan=False)]
    return out

def _x_grid(d: pd.DataFrame, pad: float = 0.25, n: int = 400) -> np.ndarray:
    xs = pd.to_numeric(d["standard_angle_abs"], errors="coerce").dropna().to_numpy()
    p  = pd.to_numeric(d["pse"], errors="coerce").dropna().to_numpy()
    w  = pd.to_numeric(d["width"], errors="coerce").dropna().to_numpy()
    if xs.size == 0 and p.size == 0: return np.linspace(-10, 10, n)
    xmin = np.nanmin([xs.min(initial=0), p.min(initial=0) - 4*np.nanmean(w) if w.size else 0])
    xmax = np.nanmax([xs.max(initial=0), p.max(initial=0) + 4*np.nanmean(w) if w.size else 0])
    rng = xmax - xmin if xmax > xmin else 10.0
    return np.linspace(X_MIN, X_MAX, n)

def _curve_from_row(x: np.ndarray, row: pd.Series) -> np.ndarray:
    return psychometric_y(
        x,
        threshold=float(row["pse"]),
        width=float(row["width"]),
        gamma=float(row.get("gamma", 0.0)),
        lam=float(row.get("lambda", 0.0)),
        eta=float(row.get("eta", 0.0)),
    )

def _aggregate_curves(x: np.ndarray, rows: List[pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate each subject's curve on x, return (median, ci_lo, ci_hi) with 68% envelope.
    """
    if not rows:
        return np.full_like(x, np.nan), np.full_like(x, np.nan), np.full_like(x, np.nan), np.full_like(x, np.nan)
    Y = np.vstack([_curve_from_row(x, r) for r in rows])
    med = np.nanmedian(Y, axis=0)
    mean = np.nanmean(Y, axis=0)
    lo  = np.nanpercentile(Y, 16, axis=0)
    hi  = np.nanpercentile(Y, 84, axis=0)
    return mean, med, lo, hi


def _mean_and_sem(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given array Y (subjects × x), return (mean, lo, hi) where lo/hi = mean ± SEM.
    Handles NaNs; SEM=0 where n_eff<=1.
    """
    Y = np.asarray(Y, dtype=float)
    mean = np.nanmean(Y, axis=0)
    n_eff = np.sum(np.isfinite(Y), axis=0)
    # ddof=1 for an unbiased sd; NaN when n<=1, we'll zero those SEMs
    sd = np.nanstd(Y, axis=0, ddof=1)
    sem = sd / np.sqrt(np.clip(n_eff, 1, None))
    sem[n_eff <= 1] = 0.0
    lo = mean - sem
    hi = mean + sem
    return mean, lo, hi


def _percentile_band(Y: np.ndarray, conf: float = 0.68):
    """
    Between-subject band: lo/hi are pointwise percentiles across subjects.
    Y: (subjects x xgrid)
    """
    Y = np.asarray(Y, dtype=float)
    mean = np.nanmean(Y, axis=0)
    qlo = (1.0 - conf) / 2.0 * 100.0
    qhi = (1.0 + conf) / 2.0 * 100.0
    lo = np.nanpercentile(Y, qlo, axis=0)
    hi = np.nanpercentile(Y, qhi, axis=0)
    return mean, lo, hi

def _bootstrap_subject_band(Y: np.ndarray, conf: float = 0.68, n_boot: int = 1000, random_state: int | None = 0):
    """
    Cluster/bootstrap over subjects: resample rows of Y, take mean per bootstrap,
    return percentiles across bootstraps. Captures uncertainty of the group mean.
    """
    Y = np.asarray(Y, dtype=float)
    S, T = Y.shape
    rng = np.random.default_rng(random_state)
    boots = np.empty((n_boot, T), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, S, size=S)      # resample subjects
        boots[b, :] = np.nanmean(Y[idx, :], axis=0)
    qlo = (1.0 - conf) / 2.0 * 100.0
    qhi = (1.0 + conf) / 2.0 * 100.0
    lo = np.nanpercentile(boots, qlo, axis=0)
    hi = np.nanpercentile(boots, qhi, axis=0)
    mean = np.nanmean(Y, axis=0)             # point estimate shown as thick line
    return mean, lo, hi

def _order_trial_types_for_panelD(dataset: str, tt_list: list[str]) -> list[str]:
    """
    Preferred order for bars on Panel D.
    - combined_cue: C-C, T-C, C-T, T-T  (matches your figure)
    - otherwise: within-cue first (sorted by cue name), then across-cue (sorted).
    """
    tts = [str(t) for t in tt_list]
    if dataset == "combined_cue":
        desired = ["BOTH-->BOTH", "ITD-->BOTH", "BOTH-->ITD", "ITD-->ITD"]
        # keep only those present, in that order; then append any leftovers
        ordered = [t for t in desired if t in tts] + [t for t in tts if t not in desired]
        return ordered

    if dataset == "single_cue":
        desired = ["ILD-->ILD", "ITD-->ILD", "ILD-->ITD", "ITD-->ITD"]
        # keep only those present, in that order; then append any leftovers
        ordered = [t for t in desired if t in tts] + [t for t in tts if t not in desired]
        return ordered

    # generic fallback
    def is_within(t: str) -> bool:
        try:
            a, b = [s.strip() for s in t.split("-->")]
            return a == b
        except Exception:
            return False

    within = sorted([t for t in tts if is_within(t)])
    across = sorted([t for t in tts if not is_within(t)])
    return [within[0] + across[0] + across[1], within[1]]


def _add_box(ax, data, pos, color, *, width=0.18, alpha=0.30, lw=1.2, z=3):
    """
    Draw a single boxplot at x=pos with matplotlib (patch_artist=True),
    colored with `color`, on top of existing points.
    """
    if data is None or len(data) == 0 or not np.isfinite(np.asarray(data, float)).any():
        return
    bp = ax.boxplot(
        [np.asarray(data, float)], positions=[pos], widths=width,
        patch_artist=True, showfliers=False, manage_ticks=False, zorder=z,
        whiskerprops=dict(linewidth=lw, color=color, alpha=0.9),
        capprops=dict(linewidth=lw, color=color, alpha=0.9),
        medianprops=dict(linewidth=lw+0.6, color=color, alpha=1.0),
        boxprops=dict(linewidth=lw, color=color)
    )
    # face color with transparency + ensure zorder on box patch
    for patch in bp["boxes"]:
        patch.set_facecolor(color); patch.set_alpha(alpha); patch.set_zorder(z)

def _add_box_simple(
    ax, data, pos, *,
    width=0.28, line_style="-", line_color="k",
    lw=1, fill=False, alpha=0.15, z=4
):
    """One boxplot at x=pos, styled by line_style (e.g., '--' for Uncertainty)."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    bp = ax.boxplot(
        [arr], positions=[pos], widths=width,
        patch_artist=True, showfliers=False, manage_ticks=False, zorder=z,
        whiskerprops=dict(linewidth=lw, color=line_color, linestyle=line_style),
        capprops=dict(linewidth=lw, color=line_color, linestyle=line_style),
        medianprops=dict(linewidth=lw+0.6, color=line_color, linestyle="-"),  # median stays solid
        boxprops=dict(linewidth=lw, edgecolor=line_color)  # we'll override style below
    )

    # --- enforce linestyle on the box patch explicitly ---
    for patch in bp["boxes"]:
        patch.set_facecolor(line_color if fill else "none")
        patch.set_alpha(alpha if fill else 1.0)
        patch.set_edgecolor(line_color)
        patch.set_linewidth(lw)
        patch.set_linestyle(line_style)   # <- this makes the rectangle dashed when needed
        patch.set_zorder(z)

    # also ensure whiskers/caps pick up linestyle (some backends need explicit set)
    for artist_name in ("whiskers", "caps"):
        for ln in bp.get(artist_name, []):
            ln.set_linestyle(line_style)
            ln.set_color(line_color)
            ln.set_linewidth(lw)
            ln.set_zorder(z)

from matplotlib import colors as mcolors

def _darken_color(color, delta: int = 20):
    """Darken an RGBA/hex/mpl color by `delta` brightness points (0–255)."""
    r, g, b, a = mcolors.to_rgba(color)
    d = delta / 255.0
    return (max(0.0, r - d), max(0.0, g - d), max(0.0, b - d), a)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False, direction="out")


def _set_int_triplet_yticks(ax, values, pad_ratio: float = 0.10, min_span: int = 2):
    """
    Set y-ticks to [low_int, 0, high_int] based on `values`.
    Ensures integers, includes 0, and avoids decimals.
    """
    import numpy as np
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        low, high = -1, 1  # sensible default
        ax.set_ylim(low, high)
        ax.set_yticks([low, 0, high]); ax.set_yticklabels([str(low), "0", str(high)])
        return

    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = vmax - vmin
    pad = pad_ratio * (span if span > 0 else 1.0)

    low = int(np.floor(vmin - pad))
    high = int(np.ceil(vmax + pad))

    # ensure 0 is strictly inside the range and ticks are integers
    if low >= 0:   low  = -max(1, high if high > 0 else 1)
    if high <= 0:  high =  max(1, -low if low < 0 else 1)

    # avoid degenerate spans
    if high - low < min_span:
        high = low + max(min_span, 2)

    ax.set_ylim(low, high)
    ax.set_yticks([low, 0, high])
    ax.set_yticklabels([str(low), "0", str(high)])



# ---------- main panel plotter ----------
def plot_panels_AB(
    df: pd.DataFrame,
    *,
    dataset: str,
    frequency: Optional[float] = None,   # required for combined_cue
    save_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (8, 3.5),
    thin_alpha: float = 0.15,
    thin_lw: float = 1.,
    thick_lw: float = 2.0,
    ci_alpha: float = 0.25,
) -> plt.Figure:
    """
    Make the two left panels:
      A. within-cue curves (thin per subject; thick median+CI per trial_type) + JND bars
      B. across-cue curves + pse_delta arrows
    """
    data = _select_experiment(df, dataset=dataset, frequency=frequency)
    if data.empty:
        raise ValueError("No rows for requested experiment.")


    # split panels
    within = data[_is_within_cue(data)].copy()
    across = data[~_is_within_cue(data)].copy()

    colmap = trialtype_colors()

    # figure with a narrow right column for Panel C

    # figure: left column = A/B, right column = C (top) and D (bottom)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 5, width_ratios=[3.6, 0.8, 1.6, 1.6, 1.6], hspace=0.22, wspace=0.35)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[:, 1])  # Panel C (ΔPSE)
    axD = fig.add_subplot(gs[:, 2])  # Panel D (JND)
    axE = fig.add_subplot(gs[:, 3])  # Panel E
    axF = fig.add_subplot(gs[:, 4])  # Panel E

    # ------------ Panel A: within-cue ------------
    if not within.empty:
        x = _x_grid(within)
        for tt, g in within.groupby("trial_type", dropna=False):
            color = colmap.get(str(tt), matplotlib.cm.get_cmap("tab20")(6))
            # thin subject curves
            x_common = np.linspace(X_MIN, X_MAX, 800)
            Ycurves, PSEs, X84s = [], [], []
            for _, row in g.iterrows():
                res_path = row.get("result_json_path", "")
                if isinstance(res_path, str) and res_path:
                    pse_i, x84_i = _plot_curve_from_result_path(
                        axA, res_path,
                        line_color=color, line_width=thin_lw, line_style="-", alpha=thin_alpha
                    )
                    try:
                        result_i = _load_result(res_path)
                        y_i = _curve_on_grid_from_result(result_i, x_common)
                        Ycurves.append(y_i)
                    except Exception:
                        pass
                    if pse_i is not None: PSEs.append(pse_i)
                    if x84_i is not None: X84s.append(x84_i)

            # ----- thick group curve (median over subjects) + 68% band -----
            if Ycurves:
                Y = np.vstack(Ycurves)  # rows = subjects, cols = x grid
                mean, lo, hi = _percentile_band(Y)
                axA.fill_between(x_common, lo, hi, color=color, alpha=ci_alpha, linewidth=0)
                axA.plot(x_common, mean, color=color, lw=thick_lw, alpha=0.95, label=short_label(str(tt)))

            # JND bar (median)
            pse_med = np.nanmean(pd.to_numeric(g["pse"], errors="coerce"))
            jnd_med = np.nanmean(pd.to_numeric(g["jnd"], errors="coerce"))
            if np.isfinite(pse_med) and np.isfinite(jnd_med):
                ybar = 0.5 - 0.035  # just under 50%
                axA.hlines(ybar, pse_med, pse_med + jnd_med, color=color, lw=5, alpha=0.9)

            # --- NEW: verticals at median PSE (→0.5) and PSE+JND (→0.84) ---
            if np.isfinite(pse_med):
                axA.vlines(pse_med, 0.0, 0.5, color="k", lw=0.8, alpha=0.1)
            if np.isfinite(pse_med) and np.isfinite(jnd_med):
                x84 = pse_med + jnd_med
                # axA.vlines(x84, 0.0, 0.84, color="k", lw=0.8, alpha=0.25)

    # decorations A
    axA.axhline(0.5, color="k", lw=0.8, alpha=0.1)
    axA.axhline(0.84, color="k", lw=0.8, alpha=0.1)
    axA.axvline(0.0, color="k", lw=0.9, ls="--", alpha=0.1)
    # axA.set_ylabel('Proportion "further right"')
    axA.set_title("A.", loc="left", fontweight="bold")
    if within["trial_type"].nunique() > 0:
        axA.legend(frameon=False, ncol=3, fontsize=9, title=None, loc="lower right")

    # --- inside plot_panels_AB, before "for tt, g in across.groupby(...)" ---
    # median within-cue JNDs per cue (strength map)
    cue_jnd_med = {}
    if not within.empty:
        for tt_wc, g_wc in within.groupby("trial_type", dropna=False):
            cue = str(tt_wc).split("-->")[0].strip()  # 'ITD', 'ILD', 'BOTH'
            cue_jnd_med[cue] = float(np.nanmedian(pd.to_numeric(g_wc["jnd"], errors="coerce")))

    # staggered y-positions for arrows to avoid overlap
    across_tts = sorted(across["trial_type"].astype(str).dropna().unique().tolist())
    arrow_y_levels = np.linspace(0.08, 0.20, num=max(2, len(across_tts)))
    arrow_y_for_tt = {tt: float(arrow_y_levels[i]) for i, tt in enumerate(across_tts)}

    # ------------ Panel B: across-cue ------------
    if not across.empty:
        x = _x_grid(across)
        for tt, g in across.groupby("trial_type", dropna=False):
            color = colmap.get(str(tt), matplotlib.cm.get_cmap("tab20")(6))
            # thin subject curves
            x_common = np.linspace(X_MIN, X_MAX, 800)
            Ycurves, PSEs, X84s = [], [], []
            for _, row in g.iterrows():
                res_path = row.get("result_json_path", "")
                if isinstance(res_path, str) and res_path:
                    pse_i, x84_i = _plot_curve_from_result_path(
                        axB, res_path,
                        line_color=color, line_width=thin_lw, line_style="-", alpha=thin_alpha
                    )
                    try:
                        result_i = _load_result(res_path)
                        y_i = _curve_on_grid_from_result(result_i, x_common)
                        Ycurves.append(y_i)
                    except Exception:
                        pass
                    if pse_i is not None: PSEs.append(pse_i)
                    if x84_i is not None: X84s.append(x84_i)

            # ----- thick group curve (median over subjects) + 68% band -----
            if Ycurves:
                Y = np.vstack(Ycurves)  # rows = subjects, cols = x grid
                mean, lo, hi = _percentile_band(Y)
                axB.fill_between(x_common, lo, hi, color=color, alpha=ci_alpha, linewidth=0)
                axB.plot(x_common, mean, color=color, lw=thick_lw, alpha=0.95, label=short_label(str(tt)))

            # --- JND bar (median) for across-cue ---
            pse_med = np.nanmean(pd.to_numeric(g["pse"], errors="coerce"))
            jnd_med = np.nanmean(pd.to_numeric(g["jnd"], errors="coerce"))
            if np.isfinite(pse_med) and np.isfinite(jnd_med):
                ybarB = 0.5 - 0.035  # just under 50%
                axB.hlines(ybarB, pse_med, pse_med + jnd_med, color=color, lw=5, alpha=0.9)

            # --- NEW: verticals at median PSE (→0.5) and PSE+JND (→0.84) ---
            if np.isfinite(pse_med):
                axB.vlines(pse_med, 0.0, 0.5, color="k", lw=0.8, alpha=0.1)
            if np.isfinite(pse_med) and np.isfinite(jnd_med):
                x84 = pse_med + jnd_med
                # axB.vlines(x84, 0.0, 0.84, color="k", lw=0.8, alpha=0.25)

            # --- ΔPSE arrow (median), one-headed and color-coded by cue strength ---
            pse_med = np.nanmedian(pd.to_numeric(g["pse"], errors="coerce"))
            ang_med = np.nanmedian(pd.to_numeric(g["standard_angle_abs"], errors="coerce"))
            if np.isfinite(pse_med) and np.isfinite(ang_med):
                # weak/strong decision from within-cue medians (larger JND = weaker)
                std_cue, cmp_cue = [s.strip() for s in str(tt).split("-->")]
                j_std = cue_jnd_med.get(std_cue, np.nan)
                j_cmp = cue_jnd_med.get(cmp_cue, np.nan)
                if np.isfinite(j_std) and np.isfinite(j_cmp):
                    arrow_color = ARROW_WEAK2STRONG if (j_std > j_cmp) else ARROW_STRONG2WEAK
                else:
                    arrow_color = "#999999"  # fallback

                y0 = arrow_y_for_tt.get(str(tt), 0.10)
                # arrow head ONLY at the PSE end
                axB.annotate(
                    "",
                    xy=(pse_med, y0), xytext=(ang_med, y0),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color=arrow_color,
                                    shrinkA=0, shrinkB=0, alpha=0.95),
                )

    # decorations B
    axB.axhline(0.5, color="k", lw=0.8, alpha=0.1)
    axB.axhline(0.84, color="k", lw=0.8, alpha=0.1)
    axB.axvline(0.0, color="k", lw=0.9, ls="--", alpha=0.1)
    # axB.set_ylabel('Proportion "further right"')
    axB.set_xlabel("Stimulus location (°)")
    axB.set_title("B.", loc="left", fontweight="bold")

    # lock x-range for both panels
    axA.set_xlim(X_MIN, X_MAX)
    axB.set_xlim(X_MIN, X_MAX)

    axA.set_ylim(0, 1)
    axB.set_ylim(0, 1)

    # ------------ Panel C: ΔPSE (across-cue), bars = subject-median, dots = per-subject ------------
    # decide weak/strong from within-cue JND medians
    weak_cue, strong_cue = None, None
    if cue_jnd_med:
        # smaller JND = stronger cue
        cues_sorted = sorted([(k, v) for k, v in cue_jnd_med.items() if np.isfinite(v)], key=lambda t: t[1])
        if len(cues_sorted) >= 2:
            strong_cue = cues_sorted[0][0]
            weak_cue = cues_sorted[-1][0]

    def _bucket_for_trialtype(tt: object) -> Optional[str]:
        try:
            a, b = [s.strip() for s in str(tt).split("-->")]
        except Exception:
            return None
        if weak_cue and strong_cue:
            if a == weak_cue and b == strong_cue:
                return "weak2strong"
            if a == strong_cue and b == weak_cue:
                return "strong2weak"
        return None

    # only across-cue rows, with bucket tag
    across_for_c = across.copy()
    across_for_c["bucket"] = across_for_c["trial_type"].map(_bucket_for_trialtype)
    across_for_c = across_for_c[across_for_c["bucket"].notna()]

    # per-subject median ΔPSE within each bucket
    subj_delta = (
        across_for_c.groupby(["subject_key", "bucket"], dropna=False)["pse_delta"]
        .median()
        .reset_index()
    )

    print(subj_delta)

    # vectors for plotting
    w2s = subj_delta.loc[subj_delta["bucket"] == "weak2strong", "pse_delta"].to_numpy()
    s2w = subj_delta.loc[subj_delta["bucket"] == "strong2weak", "pse_delta"].to_numpy()

    # bars = median over subjects (behind the dots)
    med_w2s = float(np.nanmedian(w2s)) if w2s.size else np.nan
    med_s2w = float(np.nanmedian(s2w)) if s2w.size else np.nan
    axC.bar([0, 1], [med_w2s, med_s2w],
            color=[BAR_WEAK2STRONG, BAR_STRONG2WEAK], alpha=1.,
            width=0.60, edgecolor="none", zorder=1)

    # scatter/swarm-ish dots with jitter
    rng = np.random.default_rng(0)

    def _jitter(x0, n, width=0.25):
        return x0 + (rng.random(n) - 0.5) * width

    if w2s.size:
        axC.scatter(_jitter(0.0, w2s.size), w2s, s=22, color=DOT_WEAK2STRONG,
                    edgecolors="none", zorder=2, alpha=1.)
    if s2w.size:
        axC.scatter(_jitter(1.0, s2w.size), s2w, s=22, color=DOT_STRONG2WEAK,
                    edgecolors="none", zorder=2, alpha=1.)

    # cosmetics
    axC.axhline(0.0, color="k", lw=1.0, alpha=0.6, ls=(0, (3, 3)), zorder=0)

    # x-labels as Δ{weak}-{strong} / Δ{strong}-{weak} if known; else generic
    def _abbr(cue: Optional[str]) -> str:
        m = {"ITD": "T", "ILD": "L", "BOTH": "C", "Both": "C", "combined": "C", "Combined": "C"}
        return m.get(str(cue), (str(cue)[:1] if cue else "?"))

    xlabels = [
        f"Δ{_abbr(weak_cue)}-{_abbr(strong_cue)}" if weak_cue and strong_cue else "Δ weak→strong",
        f"Δ{_abbr(strong_cue)}-{_abbr(weak_cue)}" if weak_cue and strong_cue else "Δ strong→weak",
    ]
    axC.set_xticks([0, 1])
    axC.set_xticklabels(xlabels)
    axC.set_title("C.", loc="left", fontweight="bold")
    # axC.set_ylabel("ΔPSE = PSE − standard angle (°)")
    # keep a tidy y-range
    yvals = np.r_[w2s, s2w]
    _set_int_triplet_yticks(axC, yvals)

    # ------------ Panel D: JND by trial type (subject dots + median bars) ------------
    # subject-level JND median per trial type
    jnd_subj = (
        data.groupby(["subject_key", "trial_type"], dropna=False)["jnd"]
        .median()
        .reset_index()
        .rename(columns={"jnd": "jnd_med_subj"})
    )

    # x order of categories
    tt_order = _order_trial_types_for_panelD(dataset, jnd_subj["trial_type"].unique().tolist())

    # prepare plotting vectors
    x_pos = {tt: i for i, tt in enumerate(tt_order)}
    colmap = trialtype_colors()

    # bars = median across subjects per trial type
    bars_y = []
    bars_x = []
    bars_c = []
    for tt in tt_order:
        yvals = jnd_subj.loc[jnd_subj["trial_type"] == tt, "jnd_med_subj"].to_numpy(dtype=float)
        if yvals.size == 0 or not np.isfinite(yvals).any():
            continue
        bars_x.append(x_pos[tt])
        bars_y.append(float(np.nanmedian(yvals)))
        bars_c.append(colmap.get(tt, matplotlib.cm.get_cmap("tab20")(6)))

    axD.bar(bars_x, bars_y, color=bars_c, alpha=1., width=0.70, edgecolor="none", zorder=1)

    # dots = each subject (with a little jitter)
    rng = np.random.default_rng(0)

    def _jitter(x0, n, width=0.30):
        return x0 + (rng.random(n) - 0.5) * width

    for tt in tt_order:
        ys = jnd_subj.loc[jnd_subj["trial_type"] == tt, "jnd_med_subj"].to_numpy(dtype=float)
        if ys.size == 0:
            continue
        x0 = x_pos[tt]
        base_col = colmap.get(tt, matplotlib.cm.get_cmap("tab20")(6))
        dot_col = _darken_color(base_col, delta=20)  # <- darker by 20/255
        axD.scatter(_jitter(x0, ys.size), ys, s=22,
                    color=dot_col, edgecolors="none", alpha=0.95, zorder=2)

    # cosmetics
    axD.set_xticks([x_pos[tt] for tt in tt_order])
    axD.set_xticklabels([short_label(tt) for tt in tt_order])
    axD.set_title("D.", loc="left", fontweight="bold")
    # axD.set_ylabel("JND (°)")


    # tidy y-range
    all_y = jnd_subj["jnd_med_subj"].to_numpy(dtype=float)
    _set_int_triplet_yticks(axD, all_y)
    axD.set_ylim(0, 20)

    # light baseline at 0
    axD.axhline(0, color="k", lw=0.8, alpha=0.25)

    # ------------ Shared x-ticks = mean/median standard angles per condition ------------
    tick_x, tick_lbl = [], []
    def _ticks_from(df_part: pd.DataFrame):
        nonlocal tick_x, tick_lbl
        if df_part.empty: return
        for tt, g in df_part.groupby("trial_type", dropna=False):
            ang_med = np.nanmedian(pd.to_numeric(g["standard_angle_abs"], errors="coerce"))
            if np.isfinite(ang_med) and (X_MIN <= float(ang_med) <= X_MAX):
                tick_x.append(float(ang_med))
                tick_lbl.append(f"x{short_label(str(tt)).replace('-', '')}".lower())
    _ticks_from(within); _ticks_from(across)

    # ------------ Panel E: PSE prediction errors (uncertainty vs scaling) ------------
    # bucket rows (reuse the cue strength map built earlier)
    across_for_e = across.copy()
    across_for_e["bucket"] = across_for_e["trial_type"].map(_bucket_for_trialtype)
    across_for_e = across_for_e[across_for_e["bucket"].notna()].copy()

    # subject-level *median* error per bucket
    err_subj = (
        across_for_e.groupby(["subject_key", "bucket"], dropna=False)[
            ["pse_pred_error_uncertainty", "pse_pred_error_scaling"]
        ].median()
        .reset_index()
    )

    stat, p = wilcoxon(err_subj["pse_pred_error_uncertainty"].abs(), err_subj["pse_pred_error_scaling"].abs())
    print(f"PSE prediction statistics: W = {stat:.2f}, p = {p}")

    # split by bucket
    w2s_u = err_subj.loc[err_subj["bucket"] == "weak2strong", "pse_pred_error_uncertainty"].to_numpy(float)
    w2s_s = err_subj.loc[err_subj["bucket"] == "weak2strong", "pse_pred_error_scaling"].to_numpy(float)
    s2w_u = err_subj.loc[err_subj["bucket"] == "strong2weak", "pse_pred_error_uncertainty"].to_numpy(float)
    s2w_s = err_subj.loc[err_subj["bucket"] == "strong2weak", "pse_pred_error_scaling"].to_numpy(float)

    # swarm-ish scatter with jitter
    rng = np.random.default_rng(0)

    def _j(x0, n, w=0.1):
        return x0 + (rng.random(n) - 0.5) * w

    # x-positions: 0 = Uncertainty, 1 = Scaling
    if w2s_u.size: axE.scatter(_j(0.00 - 0.2, w2s_u.size), w2s_u, s=22, color=DOT_WEAK2STRONG, edgecolors="none", alpha=0.95,
                               zorder=2)
    if s2w_u.size: axE.scatter(_j(0.00 - 0.2, s2w_u.size), s2w_u, s=22, color=DOT_STRONG2WEAK, edgecolors="none", alpha=0.95,
                               zorder=2)
    if w2s_s.size: axE.scatter(_j(1.00 + 0.2, w2s_s.size), w2s_s, s=22, color=DOT_WEAK2STRONG, edgecolors="none", alpha=0.95,
                               zorder=2)
    if s2w_s.size: axE.scatter(_j(1.00 + 0.2, s2w_s.size), s2w_s, s=22, color=DOT_STRONG2WEAK, edgecolors="none", alpha=0.95,
                               zorder=2)

    # --- Boxplots on top (two per column: weak→strong and strong→weak) ---
    off = 0.25  # horizontal offset so the two boxes per column don't overlap

    # Uncertainty column (x=0)
    # _add_box(axE, w2s_u, pos=0.00 - off, color=DOT_WEAK2STRONG)
    # _add_box(axE, s2w_u, pos=0.00 + off, color=DOT_STRONG2WEAK)
    #
    # # Scaling column (x=1)
    # _add_box(axE, w2s_s, pos=1.00 - off, color=DOT_WEAK2STRONG)
    # _add_box(axE, s2w_s, pos=1.00 + off, color=DOT_STRONG2WEAK)

    # --- Single boxes per column (combine both direction buckets) ---
    u_all = np.r_[w2s_u, s2w_u]  # Uncertainty column (x=0)
    s_all = np.r_[w2s_s, s2w_s]  # Scaling     column (x=1)

    _add_box_simple(axE, u_all, pos=0.00 + 0.1, line_style="--", line_color="k", lw=1, width=0.2)
    _add_box_simple(axE, s_all, pos=1.00 - 0.1, line_style="-", line_color="k", lw=1, width=0.2)

    # horizontal 0-lines: dashed for Uncertainty, solid for Scaling
    half = 0.42
    # axE.hlines(0.0, 0.00 - half, 0.00 + half, color="k", lw=2.0, ls=(0, (3, 3)), zorder=1)  # dashed
    # axE.hlines(0.0, 1.00 - half, 1.00 + half, color="k", lw=2.0, ls="-", zorder=1)  # solid

    # cosmetics
    axE.set_xticks([0, 1])
    axE.set_xticklabels(["Uncertainty", "Scaling"])
    axE.set_title("E.", loc="left", fontweight="bold")
    axE.set_xlim(-0.6, 1.6)
    # tidy y-range from all values
    vals = np.r_[w2s_u, s2w_u, w2s_s, s2w_s]
    _set_int_triplet_yticks(axE, vals)

    # ------------ Panel F: JND prediction errors (uncertainty vs scaling), colored by trial type ------------
    if not across.empty:
        # subject-level median errors per trial type (across-cue only)
        err_subj_f = (
            across.groupby(["subject_key", "trial_type"], dropna=False)[
                ["jnd_pred_error_uncertainty", "jnd_pred_error_scaling"]
            ].median()
            .reset_index()
        )

        stat, p = wilcoxon(err_subj_f["jnd_pred_error_uncertainty"].abs(), err_subj_f["jnd_pred_error_scaling"].abs())
        print(f"JND prediction statistics: W = {stat:.2f}, p = {p}")

        # color per trial type (same as Panels A/B)
        colmap = trialtype_colors()
        ttypes = sorted(err_subj_f["trial_type"].astype(str).unique().tolist())

        rng = np.random.default_rng(2)

        def _j(x0, n, w=0.1):  # jitter helper
            return x0 + (rng.random(n) - 0.5) * w

        # draw dots at x=0 (Uncertainty) and x=1 (Scaling) for each trial type
        for tt in ttypes:
            col = colmap.get(tt, matplotlib.cm.get_cmap("tab20")(6))

            ys_u = err_subj_f.loc[err_subj_f["trial_type"] == tt, "jnd_pred_error_uncertainty"] \
                .to_numpy(dtype=float)
            if ys_u.size:
                axF.scatter(_j(0.00 - 0.2, ys_u.size), ys_u, s=22, color=col,
                            edgecolors="none", alpha=0.95, zorder=2)

            ys_s = err_subj_f.loc[err_subj_f["trial_type"] == tt, "jnd_pred_error_scaling"] \
                .to_numpy(dtype=float)
            if ys_s.size:
                axF.scatter(_j(1.00 + 0.2, ys_s.size), ys_s, s=22, color=col,
                            edgecolors="none", alpha=0.95, zorder=2)

            # --- Boxplots per trial type on top (multiple per column) ---
            # ttypes = sorted(err_subj_f["trial_type"].astype(str).unique().tolist()) if 'err_subj_f' in locals() else []
            # nT = max(1, len(ttypes))
            # # symmetric offsets around each column center; shrink if many ttypes
            # spread = 0.12 if nT <= 4 else 0.20 if nT <= 6 else 0.16
            # offs = np.linspace(-spread, spread, nT) if nT > 1 else np.array([0.0])
            # colmap = trialtype_colors()
            #
            # for j, tt in enumerate(ttypes):
            #     col = colmap.get(tt, matplotlib.cm.get_cmap("tab20")(6))
            #     ys_u = err_subj_f.loc[err_subj_f["trial_type"] == tt, "jnd_pred_error_uncertainty"].to_numpy(float)
            #     ys_s = err_subj_f.loc[err_subj_f["trial_type"] == tt, "jnd_pred_error_scaling"].to_numpy(float)
            #     _add_box(axF, ys_u, pos=0.00 + offs[j], color=col, width=max(0.10, spread * 0.6))
            #     _add_box(axF, ys_s, pos=1.00 + offs[j], color=col, width=max(0.10, spread * 0.6))

            # --- Single boxes per column (all trial types pooled) ---
            if 'err_subj_f' in locals() and not err_subj_f.empty:
                u_all_f = err_subj_f["jnd_pred_error_uncertainty"].to_numpy(dtype=float)  # x=0
                s_all_f = err_subj_f["jnd_pred_error_scaling"].to_numpy(dtype=float)  # x=1
                _add_box_simple(axF, u_all_f, pos=0.00 + 0.1, line_style="--", line_color="k", lw=1, width=0.2)
                _add_box_simple(axF, s_all_f, pos=1.00 - 0.1, line_style="-", line_color="k", lw=1, width=0.2)

    # horizontal 0-lines: dashed for Uncertainty, solid for Scaling
    half = 0.42
    # axF.hlines(0.0, 0.00 - half, 0.00 + half, color="k", lw=2.0, ls=(0, (3, 3)), zorder=1)
    # axF.hlines(0.0, 1.00 - half, 1.00 + half, color="k", lw=2.0, ls="-", zorder=1)

    # cosmetics
    axF.set_xticks([0, 1])
    axF.set_xticklabels(["Uncertainty", "Scaling"])
    axF.set_title("F.", loc="left", fontweight="bold")
    axF.set_xlim(-0.6, 1.6)

    # tidy y-range from all values in this panel
    vals_f = np.r_[
        err_subj_f.get("jnd_pred_error_uncertainty", pd.Series([], dtype=float)).to_numpy(dtype=float),
        err_subj_f.get("jnd_pred_error_scaling", pd.Series([], dtype=float)).to_numpy(dtype=float),
    ] if 'err_subj_f' in locals() else np.array([])
    if vals_f.size and np.isfinite(vals_f).any():
        rngy = np.nanmax(np.abs(vals_f))
        pad = 0.15 * (rngy if rngy > 0 else 1.0)
        axF.set_ylim(np.nanmin(vals_f) - pad, np.nanmax(vals_f) + pad)

    # vals_f = np.r_[w2s_u_f, s2w_u_f, w2s_s_f, s2w_s_f] if 'w2s_u_f' in locals() else np.array([])
    _set_int_triplet_yticks(axF, vals_f)

    for ax in (axA, axB, axC, axD, axE, axF):
        _despine(ax)


    if tick_x:
        # deduplicate by rounding
        rk = np.round(tick_x, 6)
        _, keep = np.unique(rk, return_index=True)
        axB.set_xticks([tick_x[i] for i in sorted(keep)])
        axB.set_xticklabels([tick_lbl[i] for i in sorted(keep)])

    # Global title (experiment tag)
    title = dataset if dataset != "combined_cue" else f"combined_cue ({int(frequency)} Hz)" if frequency else "combined_cue"
    # fig.suptitle("Psychometric Test Results — " + title, y=0.98, fontsize=14, fontweight="bold")

    # y-ticks only at 0.5 and 0.84
    for ax in (axA, axB):
        ax.set_yticks([0.5, 0.84])
        ax.set_yticklabels(["0.5", "0.84"])

    if save_path:
        sp = Path(save_path); sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=300, bbox_inches="tight")

    return fig


# ---------- convenience runner ----------
def save_panels_for_experiment(
    csv_path: Path | str = DEFAULT_CSV,
    *,
    dataset: str = "combined_cue",
    frequency: Optional[float] = 1300.0,
    out_dir: Path | str = OUT_DIR,
) -> Path:
    df = load_params(csv_path)
    df = df[~(df.subject_key == "single_cue:vp_2")]
    df = df[~(df.subject_key == "single_cue:vp_4")]
    df = df[~(df.subject_key == "single_cue:vp_14")]
    df = df[~(df.subject_key == "combined_cue:vp_3")]
    df = df[~((df.subject_key == "combined_cue:vp_9") & (df.standard_center_frequency == 1300))]
    fig = plot_panels_AB(df, dataset=dataset, frequency=frequency)
    tag = f"{dataset}" + (f"_{int(frequency)}hz" if dataset == "combined_cue" and frequency is not None else "")
    out = Path(out_dir) / f"{_safe(tag)}__panels_AB.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)
    return out

if __name__ == "__main__":
    # Example: combined_cue at 1300 Hz
    p = save_panels_for_experiment(
        csv_path=DEFAULT_CSV,
        dataset="single_cue",
        # dataset="combined_cue",
        # frequency=500.0,
        out_dir="Plots"
    )

    print("Saved:", p)

