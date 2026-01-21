# Analysis/plotting.py
# Plotting based on psychometric params + model predictions.
# Input CSV: Dataframes/psychometrics_params_selected_model_pred.csv

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("MacOSX")  # for PyCharm on macOS
import matplotlib.pyplot as plt

# ----------------------------- IO & basics -----------------------------

DEFAULT_CSV = Path("Dataframes/psychometrics_params_selected_model_pred.csv")
PLOTS_DIR   = Path("Plots")

def load_model_pred(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params file not found: {p}")
    df = pd.read_csv(p, low_memory=False)

    # enforce numeric for relevant columns (robust to missing)
    num_cols = [
        "standard_center_frequency", "standard_angle_abs",
        "pse", "jnd",
        "pse_pred_uncertainty", "pse_pred_scaling",
        "jnd_pred_uncertainty", "jnd_pred_scaling",
        "pse_delta", "pse_delta_uncertainty", "pse_delta_scaling",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _safe(s: object) -> str:
    return str(s).replace(":", "-").replace("/", "-").replace(" ", "_")

# ----------------------------- colors -----------------------------

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

def _color_for(tt: str) -> tuple:
    return trialtype_colors().get(str(tt), matplotlib.cm.get_cmap("tab20")(6))

# ----------------------------- grouping -----------------------------

def _is_within_cue(df: pd.DataFrame) -> pd.Series:
    return df["standard_cue"].astype(str) == df["comparison_cue"].astype(str)

def enumerate_experiment_groups(df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Build lists of group selectors per experiment:
      - across_frequencies:    dataset=='across_frequencies', groups by (dataset, subject_key)
      - single_cue:            dataset=='single_cue',         groups by (dataset, subject_key)
      - combined_cue_500hz:    dataset=='combined_cue' & f==500,  groups by (dataset, subject_key)
      - combined_cue_1300hz:   dataset=='combined_cue' & f==1300, groups by (dataset, subject_key)
    Returns dict: exp_name -> list of selectors (each a dict with the keys needed to filter rows).
    """
    out: Dict[str, List[dict]] = {}

    # across_frequencies
    dfa = df[df["dataset"] == "across_frequencies"]
    keys = (
        dfa.groupby(["dataset", "subject_key"], dropna=False)
           .size().reset_index().drop(columns=[0])
    )
    out["across_frequencies"] = [
        {"dataset": r["dataset"], "subject_key": r["subject_key"], "frequency": None}
        for _, r in keys.iterrows()
    ]

    # single_cue
    dfs = df[df["dataset"] == "single_cue"]
    keys = (
        dfs.groupby(["dataset", "subject_key"], dropna=False)
           .size().reset_index().drop(columns=[0])
    )
    out["single_cue"] = [
        {"dataset": r["dataset"], "subject_key": r["subject_key"], "frequency": None}
        for _, r in keys.iterrows()
    ]

    # combined_cue 500
    dfc = df[(df["dataset"] == "combined_cue") & (np.isclose(pd.to_numeric(df["standard_center_frequency"], errors="coerce"), 500.0))]
    keys = (
        dfc.groupby(["dataset", "subject_key"], dropna=False)
           .size().reset_index().drop(columns=[0])
    )
    out["combined_cue_500hz"] = [
        {"dataset": r["dataset"], "subject_key": r["subject_key"], "frequency": 500.0}
        for _, r in keys.iterrows()
    ]

    # combined_cue 1300
    dfc = df[(df["dataset"] == "combined_cue") & (np.isclose(pd.to_numeric(df["standard_center_frequency"], errors="coerce"), 1300.0))]
    keys = (
        dfc.groupby(["dataset", "subject_key"], dropna=False)
           .size().reset_index().drop(columns=[0])
    )
    out["combined_cue_1300hz"] = [
        {"dataset": r["dataset"], "subject_key": r["subject_key"], "frequency": 1300.0}
        for _, r in keys.iterrows()
    ]

    return out

def _slice_group(df: pd.DataFrame, selector: dict) -> pd.DataFrame:
    """Filter rows for a single selector dict: {'dataset':..., 'subject_key':..., 'frequency': (float or None)}"""
    dset = selector["dataset"]
    subj = selector["subject_key"]
    freq = selector.get("frequency", None)

    out = df[(df["dataset"] == dset) & (df["subject_key"] == subj)].copy()
    if (dset == "combined_cue") and (freq is not None):
        out = out[np.isclose(pd.to_numeric(out["standard_center_frequency"], errors="coerce"), float(freq), equal_nan=False)]
    return out

# ----------------------------- metrics -----------------------------

def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return np.nan
    y, yhat = obs[mask], pred[mask]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

# ----------------------------- plotters -----------------------------

def plot_pred_vs_obs_group(
    df: pd.DataFrame,
    selector: dict,
    *,
    param: str = "pse",                # "pse" or "jnd"
    model: str = "uncertainty",        # "uncertainty" or "scaling"
    save_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (5.8, 5.3),
    point_size: int = 24,
) -> plt.Figure:
    """
    Scatter of predicted vs observed for one group (subject-level),
    colored by trial_type, with y=x line and R^2.
    """
    pred_col = f"{param}_pred_{model}"
    obs_col  = param
    g = _slice_group(df, selector)
    if g.empty:
        raise ValueError(f"No rows for selector={selector}")
    if pred_col not in g.columns or obs_col not in g.columns:
        raise ValueError(f"Missing columns: need '{pred_col}' and '{obs_col}'")

    x = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
    y = pd.to_numeric(g[obs_col],  errors="coerce").to_numpy()
    r2 = _r2(y, x)

    cols = [_color_for(tt) for tt in g["trial_type"].astype(str)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=point_size, c=cols, edgecolors="none", alpha=0.85)

    # identity line
    lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
    hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
    if np.isfinite(lo) and np.isfinite(hi):
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="k", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)

    ds = selector["dataset"]; subj = selector["subject_key"]; f = selector.get("frequency", None)
    title = f"{ds} | {subj}" + (f" | f={f:g} Hz" if (ds=='combined_cue' and f is not None) else "")
    ax.set_title(f"{title} — {param.upper()} ({model})   R²={r2:.3f}", pad=10)
    ax.set_xlabel(f"Predicted {param.upper()}"); ax.set_ylabel(f"Observed {param.UPPER() if hasattr(str,'UPPER') else param.upper()}")  # guard

    # legend (unique trial_types)
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for tt, c in zip(g["trial_type"].astype(str), cols):
        if tt not in uniq:
            uniq[tt] = ax.scatter([], [], s=point_size, c=[c], edgecolors="none", label=tt)
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=9, frameon=False, ncol=1)

    ax.grid(True, alpha=0.25, lw=0.7)

    if save_path:
        sp = Path(save_path); ensure_dir(sp)
        fig.savefig(sp, dpi=200, bbox_inches="tight")
    return fig

def plot_within_cue_jnd_curves_group(
    df: pd.DataFrame,
    selector: dict,
    *,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (6.2, 5.0),
) -> plt.Figure:
    """
    Lines of JND vs standard_angle_abs for within-cue rows only, colored by trial_type.
    """
    g = _slice_group(df, selector)
    g = g[_is_within_cue(g)]
    if g.empty:
        raise ValueError(f"No within-cue rows for selector={selector}")

    fig, ax = plt.subplots(figsize=figsize)
    for tt, gg in g.groupby("trial_type", dropna=False):
        d = gg.copy().sort_values("standard_angle_abs", kind="mergesort")
        ax.plot(d["standard_angle_abs"], d["jnd"], lw=2.0, marker="o", ms=4,
                color=_color_for(str(tt)), label=str(tt), alpha=0.95)

    ds = selector["dataset"]; subj = selector["subject_key"]; f = selector.get("frequency", None)
    title = f"{ds} | {subj}" + (f" | f={f:g} Hz" if (ds=='combined_cue' and f is not None) else "")
    ax.set_title(title + " — within-cue JND", pad=10)
    ax.set_xlabel("Standard angle (deg)"); ax.set_ylabel("JND (deg)")
    ax.grid(True, alpha=0.25, lw=0.7); ax.legend(frameon=False, fontsize=9)

    if save_path:
        sp = Path(save_path); ensure_dir(sp)
        fig.savefig(sp, dpi=200, bbox_inches="tight")
    return fig

# ----------------------------- batch export -----------------------------

def export_experiment_groups(
    df: pd.DataFrame,
    out_root: Path | str = PLOTS_DIR,
    *,
    which: Optional[Sequence[str]] = None,    # None -> all experiments
    do_pred_vs_obs: bool = True,
    do_within_cue_jnd: bool = True,
) -> None:
    """
    Iterate all groups per experiment and save figures:
      - PSE predicted vs observed (uncertainty & scaling)
      - JND predicted vs observed (uncertainty & scaling)
      - Within-cue JND curves
    Saved under: Plots/<experiment>/<dataset>__<subject_key>[__fXXX].png style.
    """
    out_root = Path(out_root)
    groups = enumerate_experiment_groups(df)
    if which:
        groups = {k: v for k, v in groups.items() if k in which}

    for exp_name, selectors in groups.items():
        for sel in selectors:
            ds   = sel["dataset"]
            subj = sel["subject_key"]
            freq = sel.get("frequency", None)
            base = f"{_safe(ds)}__{_safe(subj)}" + (f"__f{int(freq) if (freq is not None and float(freq).is_integer()) else str(freq).replace('.','_')}" if (ds=='combined_cue' and freq is not None) else "")
            out_dir = out_root / exp_name

            if do_pred_vs_obs:
                for param in ("pse", "jnd"):
                    for model in ("uncertainty", "scaling"):
                        sp = out_dir / f"{base}__{param}_pred_vs_obs__{model}.png"
                        try:
                            fig = plot_pred_vs_obs_group(df, sel, param=param, model=model, save_path=sp)
                            plt.close(fig)
                        except Exception as e:
                            print(f"[warn] pred_vs_obs failed {exp_name} | {sel} | {param}/{model}: {e}")

            if do_within_cue_jnd:
                sp = out_dir / f"{base}__within_cue_jnd.png"
                try:
                    fig = plot_within_cue_jnd_curves_group(df, sel, save_path=sp)
                    plt.close(fig)
                except Exception as e:
                    print(f"[warn] within_cue_jnd failed {exp_name} | {sel}: {e}")

# ----------------------------- quick demo -----------------------------

if __name__ == "__main__":
    df = load_model_pred(DEFAULT_CSV)

    # Example: export everything for all experiments
    export_experiment_groups(
        df,
        out_root="Plots",
        which=None,                 # or e.g. ["across_frequencies", "combined_cue_1300hz"]
        do_pred_vs_obs=True,
        do_within_cue_jnd=True,
    )

    # Or: single figure for a specific group
    # sel = {"dataset": "combined_cue", "subject_key": "combined_cue:vp_3", "frequency": 1300.0}
    # fig = plot_pred_vs_obs_group(df, sel, param="pse", model="uncertainty",
    #                              save_path="Plots/debug__pse_uncertainty.png")
    # plt.show()
