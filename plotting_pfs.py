# Two stacked panels with JND bars
# Top:  ITD-->ITD & BOTH-->BOTH
# Bottom: ITD-->BOTH & BOTH-->ITD
# Colors:
# "ITD-->ITD": "#FF7F0E",
# "BOTH-->BOTH": "#2ca02c",
# "ITD-->BOTH": "#98DF8A",
# "BOTH-->ITD": "#FFBB78"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.special import expit as logistic, logit as logit_fn
from scipy.stats import norm

# ----------------------------
# Config
# ----------------------------
SAVE_PATH  = "JND_within_and_across_cues.svg"
CSV_PATH   = "df_all_combined_cue_1300Hz.csv"
X_COL      = "comparison_angle_abs"
Y_COL      = "score_abs"          # 0/1
COND_COL   = "trial_type"
SUBJ_COL   = "subject"

TOP_CONDS    = ["ITD-->ITD", "BOTH-->BOTH"]
BOTTOM_CONDS = ["ITD-->BOTH", "BOTH-->ITD"]

COND_COLORS = {
    "ITD-->ITD": "#FF7F0E",
    "BOTH-->BOTH": "#2ca02c",
    "ITD-->BOTH": "#98DF8A",
    "BOTH-->ITD": "#FFBB78",
}

FIGSIZE = (10, 8)
N_X = 300

SUBJ_LINE_ALPHA = 0.25
SUBJ_LINEWIDTH  = 1.0
COND_LINEWIDTH  = 2.4
CI_ALPHA        = 0.20

REFLINE_STYLE   = dict(color="#53585F", linestyle="-", linewidth=1.0, alpha=0.1, zorder=-1000)
JND_TICK_HALF   = 0.03  # vertical tick half-length around y=0.84

# ----------------------------
# Helpers
# ----------------------------
def check_binary(series):
    vals = set(pd.Series(series).dropna().unique().tolist())
    return vals.issubset({0, 1})

def fit_glm_binom(x, y, cluster=None):
    """GLM Binomial with logit link: y ~ 1 + x. Cluster-robust SEs if cluster provided."""
    X = sm.add_constant(np.asarray(x))
    model = sm.GLM(np.asarray(y), X, family=sm.families.Binomial())
    if cluster is None:
        res = model.fit()
    else:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(cluster)})
    return res

def predict_logit_with_ci(res, xs, alpha=0.05):
    """Predict on xs with delta-method 1-α CI on probability scale."""
    Xp = sm.add_constant(np.asarray(xs))
    eta = Xp @ res.params
    cov = res.cov_params()
    var_eta = np.einsum("ij,jk,ik->i", Xp, cov, Xp)
    se_eta = np.sqrt(np.maximum(var_eta, 0.0))
    p = logistic(eta)
    se_p = se_eta * p * (1 - p)
    z = norm.ppf(1 - alpha/2.0)
    lo = np.clip(p - z * se_p, 0, 1)
    hi = np.clip(p + z * se_p, 0, 1)
    return p, lo, hi

def pse_from_params(b0, b1):
    """x at p=0.5 (PSE) for logit model: logit(p)=b0+b1*x -> x = -b0/b1."""
    if b1 == 0 or not np.isfinite(b1):
        return np.nan
    return -b0 / b1

def x_at_p_from_params(b0, b1, p):
    """x such that logistic(b0 + b1*x) = p."""
    if b1 == 0 or not np.isfinite(b1) or p <= 0 or p >= 1:
        return np.nan
    return (logit_fn(p) - b0) / b1

def draw_jnd_bar(ax, color, b0, b1, y=0.84, tick_half=JND_TICK_HALF):
    """
    Draw horizontal JND bar for pooled curve:
    JND = x(0.84) - x(0.50). Drawn at y-level (default 0.84).
    """
    x50 = x_at_p_from_params(b0, b1, 0.50)  # analytically -b0/b1
    x84 = x_at_p_from_params(b0, b1, 0.84)

    if not (np.isfinite(x50) and np.isfinite(x84)):
        return None
    x_lo, x_hi = sorted([x50, x84])
    ax.hlines(0.5 - 0.01, x_lo, x_hi, color=color, linewidth=8.0, zorder=11)
    ax.vlines(x=x_hi, ymin=0, ymax=0.84, color=color, alpha=.9, linewidth=1.0, linestyle="--")
    # small end ticks (like a bracket)
    # ax.vlines([x_lo, x_hi], y - tick_half, y + tick_half, color=color, linewidth=2.0, zorder=11)
    return (x_lo, x_hi)

def plot_conditions_on_axis(ax, df, cond_list, xlim, colors):
    """Plot spaghetti (per subject) + pooled curve+CI+PSE+JND for each condition in cond_list on ax."""
    plotted = []
    xs = np.linspace(xlim[0], xlim[1], N_X)

    # reference lines: 50% and 84% (no grid)
    ax.axhline(0.50, **REFLINE_STYLE)
    ax.axhline(0.84, **REFLINE_STYLE)

    for cond in cond_list:
        d = df[df[COND_COL] == cond].copy()
        if d.empty:
            continue

        # per-subject spaghetti
        for s, g in d.groupby(SUBJ_COL):
            try:
                res_s = fit_glm_binom(g[X_COL], g[Y_COL])
                ps, _, _ = predict_logit_with_ci(res_s, xs, alpha=0.05)
                ax.plot(xs, ps, color=colors[cond], alpha=SUBJ_LINE_ALPHA, linewidth=SUBJ_LINEWIDTH)
            except Exception:
                pass  # skip if separation / too few points

        # pooled condition curve (cluster-robust by subject)
        try:
            res_c = fit_glm_binom(d[X_COL], d[Y_COL], cluster=d[SUBJ_COL])
            p_mean, p_lo, p_hi = predict_logit_with_ci(res_c, xs, alpha=0.05)
            ax.plot(xs, p_mean, color=colors[cond], linewidth=COND_LINEWIDTH, label=cond, zorder=9)
            ax.fill_between(xs, p_lo, p_hi, color=colors[cond], alpha=CI_ALPHA, linewidth=0, zorder=8)

            # PSE (vertical dashed line)
            b0, b1 = res_c.params  # [const, slope]
            pse = pse_from_params(b0, b1)
            if np.isfinite(pse):
                ax.axvline(x=pse, ymax=.5, color=colors[cond], linestyle="-", linewidth=1.1, alpha=0.9, zorder=10)

            # JND horizontal bar at 84%: from x(50%) to x(84%)
            draw_jnd_bar(ax, colors[cond], b0, b1, y=0.5, tick_half=JND_TICK_HALF)

            plotted.append(cond)
        except Exception:
            pass

    # cosmetics
    ax.set_ylim(0, 1)
    ax.set_xlim(*xlim)
    # remove top/right spines; leave left/bottom for clarity
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return plotted

# ----------------------------
# Load & checks
# ----------------------------
df = pd.read_csv(CSV_PATH)
for col in [X_COL, Y_COL, COND_COL, SUBJ_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[[X_COL, Y_COL, COND_COL, SUBJ_COL]].dropna().copy()
if not check_binary(df[Y_COL]):
    raise ValueError(f"{Y_COL} must be binary (0/1). Found values: {sorted(df[Y_COL].unique())[:10]}")

df[SUBJ_COL] = df[SUBJ_COL].astype(str)
df[COND_COL] = df[COND_COL].astype(str)

xmin_all, xmax_all = df[X_COL].min(), df[X_COL].max()
xlim = (xmin_all, xmax_all)

# ----------------------------
# Figure with two stacked axes
# ----------------------------
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, sharey=True)

plotted_top = plot_conditions_on_axis(ax_top, df, TOP_CONDS, xlim, COND_COLORS)
ax_top.set_title("ITD→ITD & BOTH→BOTH")

plotted_bottom = plot_conditions_on_axis(ax_bottom, df, BOTTOM_CONDS, xlim, COND_COLORS)
ax_bottom.set_title("ITD→BOTH & BOTH→ITD")

# Axis labels
ax_bottom.set_xlabel(X_COL)
ax_top.set_ylabel("P(response = 1)")
ax_bottom.set_ylabel("P(response = 1)")

# Legends for conditions actually present
def dedup_legend(ax):
    h, l = ax.get_legend_handles_labels()
    seen = set(); hf, lf = [], []
    for hi, li in zip(h, l):
        if li not in seen:
            seen.add(li); hf.append(hi); lf.append(li)
    if hf:
        ax.legend(hf, lf, frameon=False, loc="best")

dedup_legend(ax_top)
dedup_legend(ax_bottom)

plt.tight_layout()
if SAVE_PATH:
    plt.savefig(SAVE_PATH, bbox_inches="tight", format="svg")
plt.show()
