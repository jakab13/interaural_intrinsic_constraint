# Analysis/qc_psychometrics.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("MacOSX")  # for PyCharm on macOS
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


# Optional: psignifit to load JSON result files (if present)
try:
    import psignifit as ps  # type: ignore
except Exception:
    ps = None  # plotting still works without points if JSON not loadable

# ---------- Config ----------
DEFAULT_PARAMS_CSV = Path("Dataframes/psychometrics_params.csv")
OUT_PASS_CSV      = Path("Dataframes/psychometrics_params_qc.csv")
OUT_EXCL_CSV      = Path("Dataframes/psychometrics_params_qc_excluded.csv")

GROUPBY_KEYS: Tuple[str, ...] = (
    "dataset",
    "subject_key",
    "trial_type",
    "standard_center_frequency",
    "standard_angle_abs",
)

# ---------- IO ----------
def load_params(path: Path | str = DEFAULT_PARAMS_CSV) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find params CSV at {p}")
    return pd.read_csv(p, low_memory=False)

def save_qc_tables(df_pass: pd.DataFrame, df_excl: pd.DataFrame) -> None:
    OUT_PASS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_pass.to_csv(OUT_PASS_CSV, index=False)
    df_excl.to_csv(OUT_EXCL_CSV, index=False)

# ---------- Derived metrics (simple) ----------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # effective dynamic range
    out["effective_range"] = 1.0 - pd.to_numeric(out.get("gamma"), errors="coerce") - pd.to_numeric(out.get("lambda"), errors="coerce")

    # slope at PSE for cumulative normal: (1 - g - l) * φ(0) / width
    phi0 = 1.0 / np.sqrt(2.0 * np.pi)
    width = pd.to_numeric(out.get("width"), errors="coerce")
    erange = pd.to_numeric(out.get("effective_range"), errors="coerce")
    out["slope_at_pse"] = (erange * phi0) / width

    # Weber-like JND ratio
    angle = pd.to_numeric(out.get("standard_angle_abs"), errors="coerce").replace(0.0, np.nan)
    jnd   = pd.to_numeric(out.get("jnd"), errors="coerce")
    out["jnd_over_angle"] = jnd / angle
    return out

# ---------- QC helpers ----------
def _robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index, dtype="float64")
    sigma = 1.4826 * mad
    return (x - med).abs() / sigma

def add_qc_flags(
    df: pd.DataFrame,
    *,
    min_trials: int = 192,
    width_min: float = 1e-6,
    gl_max: float = 0.5,
    robust_z_cut: float = 3.5,
    apply_outlier_filter: bool = True,   # <— NEW: toggle
) -> pd.DataFrame:
    out = df.copy()

    width  = pd.to_numeric(out.get("width"), errors="coerce")
    gamma  = pd.to_numeric(out.get("gamma"), errors="coerce")
    lam    = pd.to_numeric(out.get("lambda"), errors="coerce")
    ntr    = pd.to_numeric(out.get("n_trials"), errors="coerce")
    jnd    = pd.to_numeric(out.get("jnd"), errors="coerce")

    out["qc_missing_params"] = width.isna() | gamma.isna() | lam.isna()
    out["qc_width_pos"]      = width > width_min
    out["qc_guess_lapse_ok"] = (gamma.between(0, gl_max, inclusive="both")) & (lam.between(0, gl_max, inclusive="both"))
    out["qc_trials_ok"]      = ntr >= min_trials
    out["qc_jnd_pos"]        = jnd > 0

    # robust JND outlier (always compute the score; we’ll *optionally* use it)
    group_keys = ["dataset", "standard_center_frequency"]
    out["qc_jnd_rz"] = out.groupby(group_keys, dropna=False)["jnd"].transform(_robust_z)
    out["qc_jnd_not_outlier"] = (out["qc_jnd_rz"].isna()) | (out["qc_jnd_rz"] <= robust_z_cut)

    # compose pass flag (optionally include outlier check)
    checks_core = [
        ~out["qc_missing_params"],
        out["qc_width_pos"],
        out["qc_guess_lapse_ok"],
        out["qc_trials_ok"],
        out["qc_jnd_pos"],
    ]
    if apply_outlier_filter:
        checks_core.append(out["qc_jnd_not_outlier"])

    out["qc_pass"] = np.logical_and.reduce(checks_core)
    out["qc_used_outlier_filter"] = bool(apply_outlier_filter)  # for transparency
    return out


# ---------- Main QC pipeline ----------
def run_qc_pipeline(
    params_csv: Path | str = DEFAULT_PARAMS_CSV,
    *,
    drop_controls: bool = True,
    write_outputs: bool = True,
    apply_outlier_filter: bool = False,  # <— default OFF so nothing is excluded for plots
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_params(params_csv)
    if "is_control" in df.columns and drop_controls:
        df = df.loc[~df["is_control"].fillna(False)].reset_index(drop=True)

    df = add_derived_columns(df)
    df = add_qc_flags(df, apply_outlier_filter=apply_outlier_filter)

    df_pass = df.loc[df["qc_pass"]].reset_index(drop=True)
    df_excl = df.loc[~df["qc_pass"]].reset_index(drop=True)

    if write_outputs:
        save_qc_tables(df_pass, df_excl)
    return df_pass, df_excl


# ---------- Color map for trial_type ----------
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

def _get_color(tt: str, cmap_map: Dict[str, tuple]) -> tuple:
    return cmap_map.get(str(tt), matplotlib.cm.get_cmap("tab20")(6))

# ---------- Optional: load result JSON to get (x, k, n) for scatter ----------
def _load_result_json(path: str | Path):
    if ps is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return ps.Result.load_json(str(p))  # type: ignore
    except Exception:
        return None

def _plot_psignifit_colored(ax, ps_plot, res, color, *, point_size: int = 8, show_points: bool = True):
    """
    Plot a psignifit Result onto `ax` using psignifit's plotter, with robust coloring.
    Tries color kwargs first; if unsupported, recolors artists after the call.
    `point_size` is in Matplotlib 'points' units (scatter area ~ size^2).
    """
    import matplotlib.pyplot as plt

    lines_before = len(ax.lines)
    cols_before = len(ax.collections)

    # Try a few kwarg spellings across psignifit versions
    tried = [
        dict(lineColor=color, dataColor=color, CIColor=color, markerSize=point_size),
        dict(line_color=color, data_color=color, ci_color=color, marker_size=point_size),
        dict(color=color),  # some builds accept just 'color'
    ]
    called = False
    for kws in tried:
        try:
            plt.sca(ax)
            ps_plot(res, **kws)
            called = True
            break
        except TypeError:
            continue
    if not called:
        plt.sca(ax)
        ps_plot(res)

    # Post-style any newly added artists
    rgba_line = mcolors.to_rgba(color)
    rgba_ci   = (*rgba_line[:3], 0.15)

    # Lines (FIT curve / CI lines)
    for ln in ax.lines[lines_before:]:
        try:
            ln.set_color(rgba_line)
        except Exception:
            pass

    # Collections (data points, CI patches)
    for coll in ax.collections[cols_before:]:
        try:
            # Heuristic: if .get_sizes() exists, treat as points
            sizes = getattr(coll, "get_sizes", lambda: None)()
            if sizes is not None:
                if show_points:
                    sizes = np.asarray(sizes)
                    if sizes.size == 0:
                        sizes = np.array([point_size ** 2], dtype=float)
                    coll.set_sizes(np.ones_like(sizes, dtype=float) * (point_size ** 2))
                coll.set_facecolor(rgba_line)
                coll.set_edgecolor(rgba_line)
            else:
                # likely CI polygon
                coll.set_facecolor(rgba_ci)
        except Exception:
            pass


# ---------- Compute psychometric curve from params (norm + equal-asymptote) ---
def _psychometric_curve(x: np.ndarray, threshold_map: float, width: float, gamma: float, lam: float) -> np.ndarray:
    """Fallback curve if we can't load the psignifit Result."""
    # p(x) = gamma + (1 - gamma - lambda) * Phi((x - mu)/sigma)
    from math import erf, sqrt
    z = (x - float(threshold_map)) / max(float(width), 1e-12)
    # vectorized erf to avoid "only length-1 arrays can be converted..." error
    erf_vec = np.vectorize(erf)
    Phi = 0.5 * (1.0 + erf_vec(z / sqrt(2.0)))
    return gamma + (1.0 - gamma - lam) * Phi

# ---------- Subject-level multi-curve viewer ----------------------------------
def _format_list(vals, fmt="{:.2f}", sep=", "):
    vals = [v for v in vals if pd.notna(v) and np.isfinite(v)]
    if not vals:
        return "NA"
    try:
        return sep.join(fmt.format(float(v)) for v in vals)
    except Exception:
        return sep.join(str(v) for v in vals)

def _values_sorted_by_angle(df_tt: pd.DataFrame, value_col: str, vfmt="{:.2f}", sep=" | "):
    """Return only the values (sorted by angle), no 'a=...' labels."""
    a = pd.to_numeric(df_tt.get("standard_angle_abs"), errors="coerce")
    v = pd.to_numeric(df_tt.get(value_col), errors="coerce")
    ok = a.notna() & v.notna() & np.isfinite(a) & np.isfinite(v)
    if not ok.any():
        return "NA"
    tmp = pd.DataFrame({"a": a[ok], "v": v[ok]}).sort_values("a", kind="mergesort")
    return sep.join(vfmt.format(float(x)) for x in tmp["v"].tolist())

def plot_subject_psychometrics(
    df_params: pd.DataFrame,
    *,
    subject_key: str,
    dataset: Optional[str] = None,
    frequency: Optional[float] = None,
    show_points: bool = True,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (12.2, 5.3),
    point_size: int = 7,
) -> plt.Figure:
    """
    Overlay curves (left) and a transposed table (right).
    Table has trial_types as columns and rows: f (Hz), Angles, PSE, JND, eta.
    Values rows list ONLY the numbers (sorted by angle), no 'a=...' labels.
    """
    # Filter rows
    sub = df_params[df_params["subject_key"] == subject_key].copy()
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    if frequency is not None:
        sub = sub[np.isclose(pd.to_numeric(sub["standard_center_frequency"], errors="coerce"), float(frequency), equal_nan=False)]
    if sub.empty:
        raise ValueError("No rows after filtering for the requested subject/dataset/frequency.")

    # Figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[3.2, 1.4], wspace=0.30, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[0, 1]); ax_tab.axis("off")

    cmap_map = trialtype_colors()
    # stable order: freq, angle, trial_type
    sub = sub.sort_values(by=["standard_center_frequency", "standard_angle_abs", "trial_type"], kind="mergesort")
    trial_types = list(dict.fromkeys(sub["trial_type"].astype(str).tolist()))  # preserve order, dedupe
    tt_colors = {tt: _get_color(tt, cmap_map) for tt in trial_types}

    # psignifit plotter if available
    ps_plot = None
    if ps is not None:
        try:
            from psignifit import psigniplot as _psp  # type: ignore
            ps_plot = _psp.plot_psychometric_function
        except Exception:
            ps_plot = None

    # ---- plot each curve ----
    for _, row in sub.iterrows():
        tt   = str(row.get("trial_type"))
        mu   = pd.to_numeric(row.get("threshold_map"), errors="coerce")
        sig  = pd.to_numeric(row.get("width"), errors="coerce")
        gam  = pd.to_numeric(row.get("gamma"), errors="coerce")
        lam  = pd.to_numeric(row.get("lambda"), errors="coerce")
        jnd  = pd.to_numeric(row.get("jnd"), errors="coerce")
        color = tt_colors.get(tt, matplotlib.cm.get_cmap("tab20")(6))

        # Try to load the saved Result JSON
        res = None
        rjp = row.get("result_json_path")
        if isinstance(rjp, str) and rjp:
            res = _load_result_json(rjp)

        if (ps_plot is not None) and (res is not None):
            _plot_psignifit_colored(ax, ps_plot, res, color, point_size=point_size, show_points=show_points)
            # ensure legend entry once per tt
            ax.plot([], [], color=color, label=tt)
        else:
            # Fallback: manual curve + optional points if we managed to load res.data
            if not (np.isfinite(mu) and np.isfinite(sig) and np.isfinite(gam) and np.isfinite(lam)):
                continue
            x_min = x_max = None
            if (res is not None) and show_points:
                try:
                    arr = np.asarray(getattr(res, "data"))
                    if arr.size and arr.shape[1] >= 3:
                        x_min = float(np.nanmin(arr[:, 0])); x_max = float(np.nanmax(arr[:, 0]))
                        pc = arr[:, 1] / arr[:, 2]
                        ax.scatter(arr[:, 0], pc, s=point_size**2, alpha=0.6, color=color, edgecolors="none")
                except Exception:
                    pass
            if x_min is None or x_max is None:
                x_min, x_max = float(mu - 4.0 * sig), float(mu + 4.0 * sig)
                if x_min == x_max: x_min, x_max = x_min - 1.0, x_max + 1.0
            xs = np.linspace(x_min, x_max, 300)
            ys = _psychometric_curve(xs, threshold_map=float(mu), width=float(sig), gamma=float(gam), lam=float(lam))
            ax.plot(xs, ys, color=color, lw=2.0, label=tt)
            ax.axvline(mu, color=color, lw=1.0, ls="--", alpha=0.35)
            if np.isfinite(jnd):
                ax.axvline(mu + jnd, color=color, lw=1.0, ls=":", alpha=0.35)

    # Ax styling
    ax.set_xlabel("Comparison angle (deg)")
    ax.set_ylabel("Proportion correct")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, lw=0.6)
    # Legend: one per trial_type
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=9, frameon=False, ncol=1)

    # Title
    bits = [f"subject: {subject_key}"]
    if dataset is not None: bits.append(f"dataset: {dataset}")
    if frequency is not None: bits.append(f"f={frequency:g} Hz")
    ax.set_title(" | ".join(bits), pad=12)

    # ---- Build transposed table: rows=metrics, columns=trial_types ----
    tt_groups = {tt: sub[sub["trial_type"].astype(str) == tt].copy() for tt in trial_types}

    freqs_row  = ["f (Hz)"]
    angles_row = ["Ref angle"]
    pse_row    = ["PSE"]
    jnd_row    = ["JND"]

    for tt in trial_types:
        df_tt = tt_groups[tt]
        freqs  = sorted({f for f in pd.to_numeric(df_tt["standard_center_frequency"], errors="coerce").tolist() if pd.notna(f)})
        angles = sorted({a for a in pd.to_numeric(df_tt["standard_angle_abs"], errors="coerce").tolist() if pd.notna(a)})

        freqs_row.append(_format_list(freqs, fmt="{:g}", sep=", "))
        angles_row.append(_format_list(angles, fmt="{:.2f}", sep=", "))

        pse_row.append(_values_sorted_by_angle(df_tt, "threshold_map", vfmt="{:.1f}", sep=" | "))
        jnd_row.append(_values_sorted_by_angle(df_tt, "jnd",           vfmt="{:.1f}", sep=" | "))

    cellText  = [freqs_row, angles_row, pse_row, jnd_row]
    colLabels = ["metric"] + trial_types

    # Wider columns: first col a bit wider, others equal
    ncols = len(colLabels)
    first = 0.28
    restw = (1.0 - first) / max(1, ncols - 1)
    colWidths = [first] + [restw] * (ncols - 1)

    table = ax_tab.table(
        cellText=cellText,
        colLabels=colLabels,
        colWidths=colWidths,   # <--- wider columns
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.35)

    # Header style: color by trial_type, grey for 'metric'
    table[(0, 0)].set_facecolor("#f0f0f0"); table[(0, 0)].set_edgecolor("white")
    for j, tt in enumerate(trial_types, start=1):
        r, g, b, a = tt_colors[tt]
        table[(0, j)].set_facecolor((r, g, b, 0.25))
        table[(0, j)].set_edgecolor("white")

    # Light row striping
    for i in range(1, len(cellText) + 1):
        for j in range(len(colLabels)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor((0, 0, 0, 0.03))
            cell.set_edgecolor("white")

    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=200, bbox_inches="tight")

    return fig

def _safe_name(s: str) -> str:
    return str(s).replace(":", "-").replace("/", "-")

def export_all_subject_panels(
    df_params: pd.DataFrame,
    out_dir: Path | str = Path("Analysis/figures_qc"),
    *,
    include_only_pass: bool = True,
    show_points: bool = True,
    point_size: int = 7,
    progress_every: int = 25,
) -> None:
    """
    Save all QC panels into a single folder with unique filenames:
      {dataset}__{subject_key_safe}__{freqTag}.png
    Special-case: for dataset == 'across_frequencies', group by (dataset, subject_key) only,
    and filename omits the frequency tag.
    """
    df = df_params.copy()
    if include_only_pass and "qc_pass" in df.columns:
        df = df[df["qc_pass"]]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build keys with special-casing
    keys_regular = (
        df[df["dataset"] != "across_frequencies"]
        .groupby(["dataset", "subject_key", "standard_center_frequency"], dropna=False)
        .size().reset_index().drop(columns=[0])
    )
    keys_across = (
        df[df["dataset"] == "across_frequencies"]
        .groupby(["dataset", "subject_key"], dropna=False)
        .size().reset_index().drop(columns=[0])
    )
    # Add a dummy frequency column to the across set for uniform handling (None)
    if not keys_across.empty:
        keys_across["standard_center_frequency"] = np.nan

    keys = pd.concat([keys_regular, keys_across], ignore_index=True)
    keys = keys.sort_values(["dataset","subject_key","standard_center_frequency"], kind="mergesort").reset_index(drop=True)

    count = len(keys)
    for i, row in keys.iterrows():
        dataset = str(row["dataset"])
        subj    = str(row["subject_key"])
        freq    = row["standard_center_frequency"]

        # Filename pieces
        subj_tag = _safe_name(subj)
        if dataset == "across_frequencies":
            fname = f"{dataset}__{subj_tag}.png"
            freq_for_plot = None
        else:
            if pd.isna(freq):
                freq_tag = "fNA"
                freq_for_plot = None
            else:
                f = float(freq)
                freq_tag = f"f{int(f) if f.is_integer() else str(f).replace('.','_')}"
                freq_for_plot = f
            fname = f"{dataset}__{subj_tag}__{freq_tag}.png"

        save_path = out_dir / fname

        try:
            fig = plot_subject_psychometrics(
                df,
                subject_key=subj,
                dataset=dataset,
                frequency=freq_for_plot,
                show_points=show_points,
                save_path=save_path,
                point_size=point_size,
            )
            plt.close(fig)
        except Exception as e:
            print(f"[warn] failed {dataset} | {subj} | {freq}: {e}")

        if progress_every and (i + 1) % progress_every == 0:
            print(f"[export] {i+1}/{count} panels saved...")

    print(f"[export] done. saved {count} panels to {out_dir}")


import numpy as np
import pandas as pd


def filter_single_cue_subject_freq(
    df: pd.DataFrame,
    mapping: dict[str, float],
    *,
    dataset_col: str = "dataset",
    subject_key_col: str = "subject_key",
    freq_col: str = "standard_center_frequency",
    dataset_name: str = "single_cue",
    atol_hz: float = 1.0,  # tolerance for matching frequency (Hz)
) -> pd.DataFrame:
    """
    Keep only rows in the single_cue dataset that match the provided subject→frequency map.
    All non-single_cue rows are kept as-is.

    mapping example:
        {"kirke": 1400, "vp_1": 1700, ...}

    Subject matching is done on the *tail* of subject_key, so both
    'single_cue:vp_3' and 'vp_3' will match 'vp_3'.
    """
    df = df.copy()

    is_sc = (df[dataset_col] == dataset_name)

    # subject id as the last token after ":" (e.g., "single_cue:vp_3" -> "vp_3")
    subj_tail = df[subject_key_col].astype(str).str.split(":").str[-1]

    # numeric frequency
    fnum = pd.to_numeric(df[freq_col], errors="coerce")

    # build allowed mask inside single_cue
    allowed = np.zeros(len(df), dtype=bool)
    for subj, f in mapping.items():
        allowed |= (subj_tail.eq(str(subj))) & np.isfinite(fnum) & np.isclose(fnum, float(f), atol=atol_hz, rtol=0.0)

    # keep: non-single_cue OR (single_cue AND allowed)
    keep_mask = (~is_sc) | (is_sc & allowed)
    return df.loc[keep_mask].reset_index(drop=True)



# ---------- Quick CLI-ish demo ----------
if __name__ == "__main__":
    # 1) run QC (drops controls by default) and save pass/excluded CSVs
    df_pass, df_excl = run_qc_pipeline(DEFAULT_PARAMS_CSV, drop_controls=True, write_outputs=True)
    print(f"QC pass: {len(df_pass)} | excluded: {len(df_excl)}")
    # 2) Example plot for a single subject (adjust subject_key/dataset/frequency)
    # fig = plot_subject_psychometrics(df_pass, subject_key="combined_cue:vp_3", dataset="combined_cue", frequency=1300.0, show_points=True, save_path="Analysis/figures_qc/combined_cue_vp3_f1300.png")
    # plt.show()

