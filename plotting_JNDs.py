# Median bars (with percentile errorbars) behind subject-shaded points
# X: trial_type, Y: JND, hue: subject (darkest = highest median JND in ITD-->ITD)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import colorsys

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "psychometric_model_parameters_combined_cue_1300.csv"
ITD_CONDITION = "ITD-->ITD"   # ranking reference condition

# Plot styling
FIGSIZE = (2.5, 6)
JITTER_WIDTH = 0.1
POINT_SIZE = 60
EDGE_COLOR = None #"#7f7f7f"
ALPHA = 1.0

# Error bars as percentiles around the median (e.g., 25/75 like a box's IQR)
P_LOW, P_HIGH = 25, 75

# Subject-shade lightness band (narrower band => gentler steps)
# darkest (highest JND) uses LOW_L, lightest (lowest JND) uses HIGH_L
LOW_L, HIGH_L = 0.35, 0.9

# Optional: set fixed base hues per category (hex or named) to keep color identity stable.
# Leave as {} to auto-assign from matplotlib's tab10.
BASE_COLORS = {
    "ITD-->ITD": "#FF7F0E",
    "BOTH-->BOTH": "#2ca02c",
    "ITD-->BOTH": "#98DF8A",
    "BOTH-->ITD": "#FFBB78",
}

# Save figure (set to a path like "figure.pdf" or "figure.svg"); keep None to skip saving
SAVE_PATH = "JND_plot_combined_cue_1300Hz.svg"

# ----------------------------
# Helpers (HLS-based color tweaks)
# ----------------------------
def rgb_to_hls(rgb):
    r, g, b = rgb
    return colorsys.rgb_to_hls(r, g, b)  # (h, l, s)

def hls_to_rgb(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def make_gentle_shade(base_rgb, rank_idx, max_rank_idx, low_l=LOW_L, high_l=HIGH_L, sat_scale=1.0):
    """
    Map subject rank (0..max_rank_idx; 0=lowest JND, max=highest JND) to a shade of base_rgb.
    Highest JND => darkest => lower lightness.
    """
    h, l, s = rgb_to_hls(base_rgb)
    t = 0 if max_rank_idx == 0 else rank_idx / max_rank_idx  # 0..1
    L = high_l - t * (high_l - low_l)  # decreasing with rank
    S = clamp01(s * sat_scale)
    return hls_to_rgb(h, L, S)

def soften_for_background(base_rgb, target_l=0.94, target_s=0.20):
    """Very light, desaturated background fill from base hue."""
    h, l, s = rgb_to_hls(base_rgb)
    return hls_to_rgb(h, target_l, target_s)

def darker_line(base_rgb, target_l=0.30):
    """Darker line color (for error bars) consistent with base hue."""
    h, l, s = rgb_to_hls(base_rgb)
    return hls_to_rgb(h, target_l, s)

def base_color_cycle(n):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]

# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# subject column auto-detect
if "subject" in df.columns:
    subj_col = "subject"
elif "Subject" in df.columns:
    subj_col = "Subject"
else:
    raise ValueError("Could not find a subject column. Expected 'subject' or 'Subject'.")

required = {"trial_type", "JND", subj_col}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["trial_type", "JND", subj_col]).copy()

# ----------------------------
# Subject ranking by median JND in ITD_CONDITION (fallback: overall)
# ----------------------------
itd_mask = df["trial_type"] == ITD_CONDITION
if itd_mask.any():
    subj_center = (df.loc[itd_mask]
                     .groupby(subj_col)["JND"]
                     .median()
                     .sort_values(ascending=True))
else:
    subj_center = (df.groupby(subj_col)["JND"]
                     .median()
                     .sort_values(ascending=True))

subjects = list(subj_center.index)  # low -> high median JND
n_subj = len(subjects)
rank_index = {s: i for i, s in enumerate(subjects)}  # 0..n_subj-1

# ----------------------------
# Categories and base colors
# ----------------------------
# trial_types = list(df["trial_type"].unique())
trial_types = ["BOTH-->BOTH", "ITD-->BOTH", "BOTH-->ITD", "ITD-->ITD"]
x_positions = {tt: i for i, tt in enumerate(trial_types)}

if BASE_COLORS:
    base_colors = {tt: to_rgb(BASE_COLORS.get(tt, base_color_cycle(1)[0])) for tt in trial_types}
else:
    base_colors = {tt: to_rgb(c) for tt, c in zip(trial_types, base_color_cycle(len(trial_types)))}

# Precompute within-category shades per subject
cat_subject_colors = {}
for tt in trial_types:
    base = base_colors[tt]
    shades = {s: make_gentle_shade(base, rank_index[s], n_subj - 1) for s in subjects}
    cat_subject_colors[tt] = shades

# ----------------------------
# Group summaries: subject-level medians -> group median + percentile bars
# ----------------------------
subj_medians = (df.groupby(["trial_type", subj_col])["JND"]
                  .median()
                  .reset_index())

def pct(a, q):
    return np.percentile(a, q) if len(a) > 0 else np.nan

summary = (subj_medians.groupby("trial_type")["JND"]
           .agg(group_median="median",
                p_low=lambda x: pct(x, P_LOW),
                p_high=lambda x: pct(x, P_HIGH))
           .reset_index())

summary["err_low"]  = summary["group_median"] - summary["p_low"]
summary["err_high"] = summary["p_high"] - summary["group_median"]

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

# 1) Background bars at group median
bar_x = [x_positions[tt] for tt in summary["trial_type"]]
bar_h = summary["group_median"].values
# bar_colors = [soften_for_background(base_colors[tt]) for tt in summary["trial_type"]]
bar_colors = [base_colors[tt] for tt in summary["trial_type"]]

ax.bar(bar_x, bar_h, width=0.8, color=bar_colors, edgecolor="none", alpha=1.0, zorder=0)

# 2) Percentile error bars (asymmetric yerr) behind points
for x, m, lo, hi, tt in zip(bar_x, bar_h, summary["err_low"], summary["err_high"], summary["trial_type"]):
    ec = darker_line(base_colors[tt], target_l=0.30)
    ax.errorbar(x, m, yerr=[[lo], [hi]], fmt="none", ecolor="black", elinewidth=1.4, capsize=3, zorder=100)

# 3) Jittered points per subject (colored by category shade)
rng = np.random.default_rng(12345)  # reproducible jitter
for tt in trial_types:
    sub_df = df[df["trial_type"] == tt]
    x0 = x_positions[tt]
    for s in subjects:
        s_data = sub_df[sub_df[subj_col] == s]
        if s_data.empty:
            continue
        x_jit = x0 + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(s_data))
        ax.scatter(x_jit, s_data["JND"],
                   s=POINT_SIZE,
                   c=[cat_subject_colors[tt][s]],
                   edgecolors=EDGE_COLOR,
                   linewidths=0.6,
                   alpha=ALPHA,
                   zorder=2)

# Cosmetics
ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(trial_types, rotation=0)
ax.set_xlabel("trial_type")
ax.set_ylabel("JND")
ax.set_title("JNDs")
ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)

# Legend: subjects ordered by ITD ranking; colored using ITD palette (or first category if ITD missing)
legend_tt = ITD_CONDITION if ITD_CONDITION in trial_types else trial_types[0]
handles = []
for s in subjects:
    color = cat_subject_colors[legend_tt][s]
    h = plt.Line2D([0], [0], marker='o', linestyle='',
                   markersize=np.sqrt(POINT_SIZE),
                   markeredgewidth=0.6,
                   markeredgecolor=EDGE_COLOR,
                   markerfacecolor=color,
                   alpha=ALPHA,
                   label=str(s))
    handles.append(h)
# ax.legend(handles=handles,
#           title=f"Subject (ordered by median JND in {legend_tt})",
#           frameon=False, loc="best", ncol=1)

plt.tight_layout()

if SAVE_PATH:
    plt.savefig(SAVE_PATH, bbox_inches="tight", format="svg")
plt.show()
