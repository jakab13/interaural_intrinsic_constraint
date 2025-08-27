"""
psychometrics.py

Build psignifit-ready data per group and run fits.

Default input table: Dataframes/df_clean_master.csv
Required columns:
  - comparison_angle_abs (float)   # x-values
  - score_abs (int/bool)           # successes per trial (summed per x)
Nice-to-have (for grouping):
  - dataset, subject_key, trial_type, standard_center_frequency, standard_angle_abs
If comparison_angle_abs is missing but comparison_angle exists, it will be derived as abs(comparison_angle).
If score_abs is missing but 'correct' exists, it will be used as score_abs.

Notes
-----
- We intentionally keep the psignifit call minimal (norm + equal asymptote).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pathlib import Path
import pandas as pd
import numpy as np

# ---- psignifit import (old API) ---------------------------------------------
try:
    import psignifit as ps  # type: ignore
except Exception:
    ps = None  # we'll raise a clear error at call-time


# ---- Defaults ----------------------------------------------------------------

DEFAULT_CSV = Path("Dataframes/df_clean_master.csv")

# Reasonable default grouping for this project (feel free to change per call)
DEFAULT_GROUPBY = (
    "dataset",
    "subject_key",
    "trial_type",
    "standard_center_frequency",
    "standard_angle_abs",
)

# ---- Utilities ---------------------------------------------------------------

def _ensure_column(df: pd.DataFrame, col: str, source: Optional[str] = None, op: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure a column exists; optionally derive from another via op:
      - op='abs' -> df[col] = abs(df[source])
    """
    if col in df.columns:
        return df
    if source is not None and source in df.columns:
        if op == "abs":
            out = df.copy()
            out[col] = pd.to_numeric(out[source], errors="coerce").abs()
            return out
    return df


def _ensure_ps_columns(
    df: pd.DataFrame,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
) -> pd.DataFrame:
    """Make sure x_col and score_col exist (derive if possible)."""
    out = df.copy()
    # x: derive from comparison_angle if needed
    out = _ensure_column(out, x_col, source="comparison_angle", op="abs")
    # score: fallback to 'correct' if 'score_abs' missing
    if score_col not in out.columns and "correct" in out.columns:
        out = out.rename(columns={"correct": score_col})
    if x_col not in out.columns:
        raise ValueError(
            f"Missing '{x_col}' and could not derive it. "
            "Provide comparison_angle_abs or comparison_angle to derive."
        )
    if score_col not in out.columns:
        raise ValueError(
            f"Missing '{score_col}' and no 'correct' column to fall back to. "
            "Please provide a binary score column."
        )
    return out


def load_clean(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Load the cleaned master CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean master not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def make_psignifit_input(
    g: pd.DataFrame,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
) -> pd.DataFrame:
    """
    Build a psignifit input table with columns:
      [x_col, 'score_abs', 'n_total']
    """
    if x_col not in g.columns or score_col not in g.columns:
        raise ValueError(f"Group missing columns: need '{x_col}' and '{score_col}'. Have: {list(g.columns)}")

    # sum successes and count trials per x level
    agg = (
        g.groupby(x_col, as_index=False)[score_col]
         .agg(score_abs="sum")
         .merge(
            g.groupby(x_col, as_index=False).size().rename(columns={"size": "n_total"}),
            on=x_col,
            how="left"
         )
         .sort_values(x_col, kind="mergesort")
         .reset_index(drop=True)
    )

    # ensure integer counts
    agg["n_total"] = agg["n_total"].astype(int)
    agg["score_abs"] = agg["score_abs"].astype(int)
    return agg[[x_col, "score_abs", "n_total"]]


@dataclass
class FitRecord:
    keys: Mapping[str, object]       # the group identifiers
    data_table: pd.DataFrame         # the Nx3 table (x, k, n)
    result: object                   # psignifit result object (opaque for now)
    ok: bool
    message: str = ""


def fit_group_with_psignifit(
    g: pd.DataFrame,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
    psignifit_kwargs: Optional[dict] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Convert a grouped slice to psignifit input and run ps.psignifit(...).
    Returns (data_table, result_object).
    """
    if ps is None:
        raise ImportError(
            "psignifit is not available in this environment. "
            "Please install/import your psignifit version and retry."
        )

    data_table = make_psignifit_input(g, x_col=x_col, score_col=score_col)
    kwargs = dict(sigmoid="norm", experiment_type="equal asymptote")
    if psignifit_kwargs:
        kwargs.update(psignifit_kwargs)
    res = ps.psignifit(data_table.values, **kwargs)
    return data_table, res


def iter_group_fits(
    df: pd.DataFrame,
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
    psignifit_kwargs: Optional[dict] = None,
) -> List[FitRecord]:
    """
    Iterate over groups, build psignifit input, and fit.
    """
    df = _ensure_ps_columns(df, x_col=x_col, score_col=score_col)

    records: List[FitRecord] = []
    for keys, g in df.groupby(list(groupby), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keymap = dict(zip(groupby, keys))

        try:
            data_table, res = fit_group_with_psignifit(
                g, x_col=x_col, score_col=score_col, psignifit_kwargs=psignifit_kwargs
            )
            records.append(FitRecord(keys=keymap, data_table=data_table, result=res, ok=True))
        except Exception as e:
            records.append(FitRecord(keys=keymap, data_table=pd.DataFrame(), result=None, ok=False, message=str(e)))
    return records


def summarize_fit_records(records: List[FitRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        n_trials = int(r.data_table["n_total"].sum()) if not r.data_table.empty else 0
        rows.append({**r.keys, "ok": r.ok, "message": r.message, "n_trials": n_trials})
    return pd.DataFrame(rows)


def run_psychometrics(
    clean_csv: Path | str = DEFAULT_CSV,
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
    psignifit_kwargs: Optional[dict] = None,
) -> Tuple[List[FitRecord], pd.DataFrame]:
    df = load_clean(clean_csv)
    recs = iter_group_fits(
        df, groupby=groupby, x_col=x_col, score_col=score_col, psignifit_kwargs=psignifit_kwargs
    )
    summary = summarize_fit_records(recs)
    return recs, summary


def sample_groups(
    df: pd.DataFrame,
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    n: int = 2,
    random_state: Optional[int] = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys_all = (
        df.groupby(list(groupby), dropna=False)
          .size()
          .reset_index()
          .rename(columns={0: "n"})
    )
    if keys_all.empty:
        raise ValueError("No groups found with the provided grouping.")
    n_pick = min(n, len(keys_all))
    keys = keys_all.sample(n=n_pick, random_state=random_state).drop(columns=["n"])
    df_small = df.merge(keys, on=list(groupby), how="inner")
    return df_small, keys


def run_psychometrics_sampled(
    clean_csv: Path | str = DEFAULT_CSV,
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    n_groups: int = 2,
    random_state: Optional[int] = 0,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
    psignifit_kwargs: Optional[dict] = None,
):
    df = load_clean(clean_csv)
    df = _ensure_ps_columns(df, x_col=x_col, score_col=score_col)
    df_small, keys = sample_groups(df, groupby=groupby, n=n_groups, random_state=random_state)
    records = iter_group_fits(df_small, groupby=groupby, x_col=x_col, score_col=score_col, psignifit_kwargs=psignifit_kwargs)
    summary = summarize_fit_records(records)
    return records, summary, keys, df_small


# -------- Robust extraction (tailored to your Result object) ------------------

def _get_attr_or_key(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _flatten_conf_intervals(ci_dict, levels=("0.95", "0.68")) -> dict:
    out = {}
    if not isinstance(ci_dict, dict):
        return out
    for pname, levelmap in ci_dict.items():
        if not isinstance(levelmap, dict):
            continue
        for lev in levels:
            if lev in levelmap and isinstance(levelmap[lev], (list, tuple)) and len(levelmap[lev]) == 2:
                lo, hi = levelmap[lev]
                tag = lev.replace(".", "")
                out[f"{pname}_ci{tag}_lo"] = float(lo)
                out[f"{pname}_ci{tag}_hi"] = float(hi)
    return out


def extract_psignifit_params_specific(res, prefer: str = "auto", ci_levels=("0.95", "0.68")) -> dict:
    """
    Extracts:
      - threshold_map (MAP['threshold'])
      - threshold_mean (MEAN['threshold'])
      - width, gamma, lambda, eta   (from preferred estimator for convenience)
      - JND = res.threshold(0.84) - threshold_map
      - CI columns flattened, trial totals from res.data
    """
    out = {}

    cfg = _get_attr_or_key(res, "configuration", {})
    est_type = _get_attr_or_key(cfg, "estimate_type", None)
    if prefer == "auto":
        prefer = (est_type or "MAP").upper()
    prefer = prefer.upper()

    p_map  = _get_attr_or_key(res, "parameters_estimate_MAP", {}) or {}
    p_mean = _get_attr_or_key(res, "parameters_estimate_mean", {}) or {}

    # main thresholds
    if "threshold" in p_map:
        out["threshold_map"] = float(p_map["threshold"])
    if "threshold" in p_mean:
        out["threshold_mean"] = float(p_mean["threshold"])

    # convenience: other params from preferred set (kept like before)
    def pick(name):
        if prefer == "MAP":
            return p_map.get(name, p_mean.get(name, np.nan))
        else:
            return p_mean.get(name, p_map.get(name, np.nan))

    for name in ("width", "gamma", "lambda", "eta"):
        val = pick(name)
        if val is not None:
            out[name] = float(val)

    # confidence intervals
    ci = _get_attr_or_key(res, "confidence_intervals", {})
    out.update(_flatten_conf_intervals(ci, levels=tuple(ci_levels)))

    # totals from data
    data = _get_attr_or_key(res, "data", None)
    if data is not None:
        try:
            arr = np.asarray(data)
            out["n_points"]  = int(arr.shape[0])
            out["n_success"] = int(arr[:, 1].sum())
            out["n_trials"]  = int(arr[:, 2].sum())
        except Exception:
            pass

    # JND: use MAP threshold as PSE baseline
    try:
        pse_map = out.get("threshold_map", np.nan)
        x84 = res.threshold(0.84, return_ci=False, unscaled=False)
        if np.isfinite(pse_map) and x84 is not None:
            out["jnd"] = float(x84) - float(pse_map)
            # 95% CI via width scaling
            w = out.get("width", np.nan)
            lo = out.get("width_ci095_lo", np.nan)
            hi = out.get("width_ci095_hi", np.nan)
            if np.isfinite(w) and w != 0 and np.isfinite(lo) and np.isfinite(hi):
                out["jnd_ci095_lo"] = float(lo / w * out["jnd"])
                out["jnd_ci095_hi"] = float(hi / w * out["jnd"])
    except Exception:
        pass

    out["estimator_prefer"] = prefer
    return out


def quick_params_table(records: list) -> pd.DataFrame:
    """
    Build a tidy parameters table from FitRecord list.
    """
    rows = []
    for r in records:
        base = dict(r.keys)
        n_trials = int(r.data_table["n_total"].sum()) if not r.data_table.empty else 0
        base.update({"ok": bool(r.ok), "n_trials": n_trials})

        if r.ok and r.result is not None:
            pars = extract_psignifit_params_specific(r.result, prefer="auto", ci_levels=("0.95", "0.68"))
            base.update(pars)
        else:
            for k in ("threshold_map", "threshold_mean", "width", "gamma", "lambda", "eta"):
                base.setdefault(k, np.nan)
            base.setdefault("estimator_prefer", "NA")

        rows.append(base)

    dfp = pd.DataFrame(rows)

    front = [c for c in [
        "dataset", "subject_key", "trial_type", "standard_center_frequency", "standard_angle_abs",
        "ok", "n_trials",
        "threshold_map", "threshold_mean", "width", "gamma", "lambda", "eta",
        "jnd", "jnd_ci095_lo", "jnd_ci095_hi",
        "threshold_ci095_lo", "threshold_ci095_hi",
        "width_ci095_lo", "width_ci095_hi",
        "estimator_prefer",
    ] if c in dfp.columns]
    rest = [c for c in dfp.columns if c not in front]
    return dfp.loc[:, front + rest]


def run_psychometrics_sampled_with_params(
    clean_csv: Path | str = DEFAULT_CSV,
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    n_groups: int = 2,
    random_state: Optional[int] = 0,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
    psignifit_kwargs: Optional[dict] = None,
):
    records, summary, keys, df_small = run_psychometrics_sampled(
        clean_csv=clean_csv,
        groupby=groupby,
        n_groups=n_groups,
        random_state=random_state,
        x_col=x_col,
        score_col=score_col,
        psignifit_kwargs=psignifit_kwargs,
    )
    params = quick_params_table(records)
    return records, summary, params, keys, df_small


# ==================== Group selection + plotting utilities ====================

import matplotlib
matplotlib.use('MacOSX')  # for interactive plotting on macOS

import re
from typing import Mapping, Sequence, Optional, Tuple
from scipy.stats import norm  # noqa: F401

PSYCH_ROOT   = Path("Psychometrics")
RESULTS_DIR  = PSYCH_ROOT / "results_json"
FIGURES_DIR  = PSYCH_ROOT / "figures"

def _safe_subject_key(s: str) -> str:
    return str(s).replace(":", "-")

def _slugify_trial_type(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")

def _format_number_for_token(x: float, tol: float = 1e-9) -> str:
    """5.0 -> '5', 8.5 -> '8_5' for filesystem safety."""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    r = round(xf)
    if abs(xf - r) <= tol:
        return str(int(r))
    return str(xf).replace(".", "_")

def _format_angle_token_2dec(x) -> str:
    """For filenames: 5   -> '5_00', 8.5 -> '8_50'."""
    try:
        return f"{float(x):.2f}".replace(".", "_")
    except Exception:
        return str(x)

def _format_angle_label_2dec(x) -> str:
    """For titles: 5   -> '5.00', 8.5 -> '8.50'."""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def build_group_paths(
    keys: Mapping[str, object],
    include_freq: bool = True,
) -> Tuple[Path, Path]:
    """
    New structure:
      Psychometrics/results_json/{dataset}/{subject_key_safe}/{f{freq}}/{trial_type_slug}/a{angle}/result.json
      Psychometrics/figures/{dataset}/{subject_key_safe}/{f{freq}}/{trial_type_slug}/a{angle}.png
    """
    dataset  = str(keys.get("dataset", "unknown"))
    subj     = _safe_subject_key(keys.get("subject_key", "unknown"))
    ttype    = _slugify_trial_type(keys.get("trial_type", "unknown"))

    # frequency folder (always present; 'fNA' if missing)
    freq = keys.get("standard_center_frequency", None)
    if include_freq:
        if freq is None or (isinstance(freq, float) and np.isnan(freq)):
            fdir = "fNA"
        else:
            fdir = f"f{_format_number_for_token(freq)}"
    else:
        fdir = "fNA"

    # angle folder
    angle = keys.get("standard_angle_abs", None)
    aseg = (
        f"a{_format_angle_token_2dec(angle)}"
        if angle is not None and not (isinstance(angle, float) and np.isnan(angle))
        else "aNA"
    )

    base_rel = Path(dataset) / subj / fdir / ttype / aseg
    result_json = (RESULTS_DIR / base_rel / "result.json")
    figure_png  = (FIGURES_DIR  / base_rel).with_suffix(".png")
    return result_json, figure_png


# ---- cues and group metadata -------------------------------------------------

_ARROW_SPLIT = re.compile(r"\s*(?:-+>|→|⟶|⇒)\s*")

def parse_cues_from_trial_type(trial_type: object) -> tuple[Optional[str], Optional[str]]:
    """Return (standard_cue, comparison_cue) parsed from strings like 'ILD-->ILD'."""
    if trial_type is None or (isinstance(trial_type, float) and np.isnan(trial_type)):
        return None, None
    s = str(trial_type).strip()
    if not s:
        return None, None
    parts = _ARROW_SPLIT.split(s)
    if len(parts) >= 2:
        return parts[0].strip() or None, parts[1].strip() or None
    return s, None


def build_group_metadata(
    df: pd.DataFrame,
    groupby: Sequence[str],
) -> pd.DataFrame:
    base = (
        df.groupby(list(groupby), dropna=False)
          .size()
          .reset_index()
          .drop(columns=[0])
    )
    if "is_control" in df.columns:
        ctrl = (
            df.groupby(list(groupby), dropna=False)["is_control"]
              .max()
              .reset_index(name="is_control")
        )
        meta = base.merge(ctrl, on=list(groupby), how="left")
        meta["is_control"] = meta["is_control"].astype("boolean")
    else:
        meta = base.copy()
        meta["is_control"] = pd.Series([pd.NA] * len(meta), dtype="boolean")

    if "trial_type" in groupby or "trial_type" in meta.columns:
        std_cues, cmp_cues = [], []
        for tt in meta["trial_type"]:
            s, c = parse_cues_from_trial_type(tt)
            std_cues.append(s)
            cmp_cues.append(c)
        meta["standard_cue"] = std_cues
        meta["comparison_cue"] = cmp_cues
    else:
        meta["standard_cue"] = pd.NA
        meta["comparison_cue"] = pd.NA

    return meta


# ---- PSE/JND for annotation --------------------------------------------------

def compute_pse_jnd_from_result(res, p_target: float = 0.84):
    """
    PSE = threshold_map = parameters_estimate_MAP['threshold']
    JND = res.threshold(p_target, ...) - PSE
    """
    try:
        p_map = _get_attr_or_key(res, "parameters_estimate_MAP", {}) or {}
        pse = p_map.get("threshold", None)
        if pse is None or not np.isfinite(pse):
            return None, None
        x_target = res.threshold(p_target, return_ci=False, unscaled=False)
        if x_target is None:
            return None, None
        jnd = float(x_target) - float(pse)
        return float(pse), float(jnd)
    except Exception:
        return None, None


# ---- IO + plotting -----------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def save_result_json(res, path: Path) -> Path:
    ensure_dir(path)
    try:
        res.save_json(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to save result JSON at {path}: {e}")
    return path

def load_result_json(path: Path):
    if not path.exists():
        return None
    try:
        return ps.Result.load_json(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load result JSON at {path}: {e}")

def plot_psychometric_from_result(
    result_obj: object,
    ax=None,
    figsize: tuple[float, float] = (6, 4),
    **plot_kwargs,
):
    if ps is None:
        raise ImportError("psignifit is not available. Please install/import it first.")
    try:
        import matplotlib.pyplot as plt
        from psignifit import psigniplot  # type: ignore
    except Exception as e:
        raise ImportError("Could not import matplotlib and psignifit.psigniplot") from e

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    import matplotlib.pyplot as plt
    plt.sca(ax)
    psigniplot.plot_psychometric_function(result_obj, **plot_kwargs)
    return ax


def plot_psychometric_with_annotations(
    result_obj,
    selector: Mapping[str, object],
    ax=None,
    figsize=(6, 4),
    **plot_kwargs,
):
    ax = plot_psychometric_from_result(result_obj, ax=ax, figsize=figsize, **plot_kwargs)

    # Title (angle 2-dec; pad to avoid overlap)
    parts = []
    for k in ("dataset", "subject_key", "trial_type"):
        if k in selector:
            parts.append(str(selector[k]))
    if "standard_center_frequency" in selector and selector["standard_center_frequency"] is not None:
        parts.append(f"f={_format_number_for_token(selector['standard_center_frequency'])}")
    if "standard_angle_abs" in selector and selector["standard_angle_abs"] is not None:
        parts.append(f"a={_format_angle_label_2dec(selector['standard_angle_abs'])}")

    title = " | ".join(parts) if parts else "Psychometric fit"
    import matplotlib.pyplot as plt
    ax.set_title(title, pad=16)

    # PSE & JND box
    pse, jnd = compute_pse_jnd_from_result(result_obj, p_target=0.84)
    lines = []
    if pse is not None:
        lines.append(f"PSE = {pse:.3f}")
    if jnd is not None:
        lines.append(f"JND (84%) = {jnd:.3f}")
    if lines:
        ax.text(
            0.03, 0.97, "\n".join(lines),
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.12, ec="none")
        )
    return ax


def save_figure(ax, path: Path, dpi: int = 300) -> Path:
    ensure_dir(path)
    import matplotlib.pyplot as plt
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def list_groups(df: pd.DataFrame, groupby: Sequence[str] = DEFAULT_GROUPBY) -> pd.DataFrame:
    return (
        df.groupby(list(groupby), dropna=False)
          .size()
          .reset_index(name="n_rows")
          .sort_values("n_rows", ascending=False, kind="mergesort")
          .reset_index(drop=True)
    )


def _slice_group(df: pd.DataFrame, selector: Mapping[str, object], groupby: Sequence[str]) -> pd.DataFrame:
    missing = [k for k in groupby if k not in selector]
    if missing:
        raise ValueError(f"Selector missing keys for groupby: {missing}")
    mask = pd.Series(True, index=df.index)
    for k in groupby:
        mask &= (df[k] == selector[k])
    g = df.loc[mask]
    if g.empty:
        raise ValueError(f"No rows found for selector={selector}")
    return g


def load_or_fit_plot_and_save_group(
    df: pd.DataFrame,
    selector: Mapping[str, object],
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    x_col: str = "comparison_angle_abs",
    score_col: str = "score_abs",
    psignifit_kwargs: Optional[dict] = None,
    overwrite: bool = False,
    overwrite_fig: Optional[bool] = None,
    include_freq_in_path: bool = True,
    save_fig: bool = True,
    fig_dpi: int = 300,
):
    """
    For a single group:
      - if JSON exists and not overwrite: load it; else fit & save JSON
      - if PNG exists and not (overwrite_fig or overwrite): skip plotting
    """
    if overwrite_fig is None:
        overwrite_fig = overwrite

    json_path, fig_path = build_group_paths(selector, include_freq=include_freq_in_path)

    # JSON: load or fit+save
    result_obj = None
    data_table = None
    if json_path.exists() and not overwrite:
        result_obj = load_result_json(json_path)
    else:
        g = _slice_group(df, selector=selector, groupby=groupby)
        data_table, result_obj = fit_group_with_psignifit(
            g, x_col=x_col, score_col=score_col, psignifit_kwargs=psignifit_kwargs
        )
        save_result_json(result_obj, json_path)

    # FIGURE: skip if already exists and not overwriting
    fp = None
    if save_fig:
        if fig_path.exists() and not overwrite_fig:
            fp = fig_path
        else:
            ax = plot_psychometric_with_annotations(result_obj, selector)
            fp = save_figure(ax, fig_path, dpi=fig_dpi)

    return result_obj, data_table, json_path, fp


# ======================= Batch runner: all groups =============================

def _result_to_data_table(res, xcol_name: str = "x") -> pd.DataFrame:
    arr = np.asarray(_get_attr_or_key(res, "data", []))
    if arr is None or arr.size == 0:
        return pd.DataFrame(columns=[xcol_name, "score_abs", "n_total"])
    df = pd.DataFrame(arr, columns=[xcol_name, "score_abs", "n_total"])
    df["n_total"] = df["n_total"].astype(int)
    df["score_abs"] = df["score_abs"].astype(int)
    return df[[xcol_name, "score_abs", "n_total"]]


def run_and_persist_all_groups(
    clean_csv: Path | str = DEFAULT_CSV,
    groupby: Sequence[str] = DEFAULT_GROUPBY,
    overwrite: bool = False,
    include_freq_in_path: bool = True,
    save_fig: bool = True,
    fig_dpi: int = 300,
    shuffle: bool = False,
    limit: Optional[int] = None,
    psignifit_kwargs: Optional[dict] = None,
    params_out: Path | str = Path("Dataframes/psychometrics_params.csv"),
    progress_every: int = 50,
) -> pd.DataFrame:
    """
    Iterate directly over df.groupby(...), saving result JSON & figures per group,
    and build a complete params table (incl. is_control, standard_cue, comparison_cue).
    """
    df = load_clean(clean_csv)

    g_iter = df.groupby(list(groupby), dropna=False)
    key_df = g_iter.size().reset_index(name="n_rows")
    if shuffle:
        key_df = key_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    if limit is not None:
        key_df = key_df.head(int(limit))

    records: List[FitRecord] = []
    path_rows: List[dict] = []
    errors: List[tuple] = []

    for i, row in key_df.iterrows():
        selector = row[list(groupby)].to_dict()

        try:
            res, data_tbl, json_path, fig_path = load_or_fit_plot_and_save_group(
                df,
                selector=selector,
                groupby=groupby,
                overwrite=overwrite,
                include_freq_in_path=include_freq_in_path,
                save_fig=save_fig,
                fig_dpi=fig_dpi,
                psignifit_kwargs=psignifit_kwargs,
            )
            if data_tbl is None or data_tbl.empty:
                data_tbl = _result_to_data_table(res)

            records.append(FitRecord(keys=selector, data_table=data_tbl, result=res, ok=True))
            path_rows.append({**selector, "result_json_path": str(json_path), "figure_png_path": str(fig_path)})

        except Exception as e:
            errors.append((selector, str(e)))
            records.append(FitRecord(keys=selector, data_table=pd.DataFrame(), result=None, ok=False, message=str(e)))
            path_rows.append({**selector, "result_json_path": None, "figure_png_path": None})

        if progress_every and (i + 1) % progress_every == 0:
            print(f"[batch] processed {i+1}/{len(key_df)} groups...")

    # params from records
    params = quick_params_table(records)

    # attach paths
    df_paths = pd.DataFrame(path_rows)
    params = params.merge(df_paths, on=list(groupby), how="left")

    # attach metadata from the clean DF (is_control + cues)
    meta = build_group_metadata(df, groupby=groupby)
    params = params.merge(
        meta[list(groupby) + ["is_control", "standard_cue", "comparison_cue"]],
        on=list(groupby), how="left"
    )

    # estimator selection ONLY on hetero ref row; no reference_threshold
    params = add_reference_estimator_column_only(params)

    # tidy column order
    front = [c for c in [
        *groupby,
        "is_control", "standard_cue", "comparison_cue",
        "ok", "n_trials",
        "threshold_map", "threshold_mean", "width", "gamma", "lambda", "eta",
        "jnd", "jnd_ci095_lo", "jnd_ci095_hi",
        "threshold_ci095_lo", "threshold_ci095_hi",
        "width_ci095_lo", "width_ci095_hi",
        "result_json_path", "figure_png_path",
        "reference_estimator",
    ] if c in params.columns]
    rest = [c for c in params.columns if c not in front]
    params = params.loc[:, front + rest]

    # save
    params_out = Path(params_out)
    params_out.parent.mkdir(parents=True, exist_ok=True)
    params.to_csv(params_out, index=False)
    try:
        params.to_parquet(params_out.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    if errors:
        print(f"[batch] completed with {len(errors)} error(s). Showing first 3:")
        for sel, msg in errors[:3]:
            print("  -", sel, "->", msg)

    print(f"[batch] saved params table to: {params_out} (rows={len(params)})")
    return params


# ==================== Reference estimator (no reference_threshold) ============

def add_reference_estimator_column_only(
    params: pd.DataFrame,
    angles: Sequence[float] = (5.0, 8.0, 10.0),
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Add only:
      - reference_estimator in {'MAP','MEAN'} filled ONLY on the heterogeneous
        reference row per (dataset, subject_key, standard_center_frequency).

    Logic:
      * For each (dataset, subject_key, standard_center_frequency), find the heterogeneous trial
        (standard_cue != comparison_cue) with standard_angle_abs ≈ {5,8,10}; pick one (max n_trials).
      * Consider its {threshold_map, threshold_mean}.
      * Look at target rows: same key, where standard_cue == (that hetero row's comparison_cue).
        Choose the estimator whose threshold is closest, on average, to the targets' standard_angle_abs.
      * Write the chosen estimator label into 'reference_estimator' ONLY on the hetero reference row.
    """
    df = params.copy()
    df["reference_estimator"] = pd.Series([pd.NA] * len(df), dtype="string")

    # numeric angle + hetero
    ang = pd.to_numeric(df.get("standard_angle_abs"), errors="coerce")
    is_hetero = df["standard_cue"].astype(str) != df["comparison_cue"].astype(str)

    # ≈ {5,8,10}
    is_ref_angle = False
    for a in angles:
        is_ref_angle = is_ref_angle | (ang.sub(float(a)).abs() <= tol)

    # candidates
    key_base = ["dataset", "subject_key", "standard_center_frequency"]
    cand = df[is_hetero & is_ref_angle].copy()

    if "n_trials" in cand.columns:
        cand = cand.sort_values(key_base + ["n_trials"], ascending=[True, True, True, False])
    refs = cand.drop_duplicates(subset=key_base, keep="first")

    for _, ref in refs.iterrows():
        key_mask = (
            (df["dataset"] == ref["dataset"]) &
            (df["subject_key"] == ref["subject_key"]) &
            (pd.to_numeric(df["standard_center_frequency"], errors="coerce")
             == pd.to_numeric(ref["standard_center_frequency"], errors="coerce"))
        )

        # targets: comparison cue of hetero becomes standard cue
        targets_mask = key_mask & (df["standard_cue"].astype(str) == str(ref["comparison_cue"]))
        target_angles = pd.to_numeric(df.loc[targets_mask, "standard_angle_abs"], errors="coerce").dropna()
        if target_angles.empty:
            continue

        thr_map  = pd.to_numeric(pd.Series(ref.get("threshold_map")), errors="coerce").iloc[0]
        thr_mean = pd.to_numeric(pd.Series(ref.get("threshold_mean")), errors="coerce").iloc[0]

        cand_vals = np.array([thr_map, thr_mean], dtype=float)
        if not np.isfinite(cand_vals).any():
            continue

        diffs = []
        for val in cand_vals:
            if np.isfinite(val):
                diffs.append(np.mean(np.abs(target_angles.to_numpy(dtype=float) - float(val))))
            else:
                diffs.append(np.inf)
        diffs = np.array(diffs)

        # choose smallest; tie-break in favor of MAP
        idx = int(np.argmin(diffs))
        chosen_label = "MAP" if idx == 0 else "MEAN"

        # set ONLY on the hetero reference row
        df.loc[ref.name, "reference_estimator"] = chosen_label

    return df
