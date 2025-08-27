# Analysis/model_prep.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

DEFAULT_SELECTED = Path("Dataframes/psychometrics_params_selected.csv")

# ---------- IO ----------
def load_selected_params(path: Path | str = DEFAULT_SELECTED) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Selected params CSV not found at {p}")
    return pd.read_csv(p, low_memory=False)

# ---------- renaming: threshold* -> pse* (lower case) ----------
def _build_threshold_rename_map(cols: pd.Index) -> Dict[str, str]:
    have_map = "threshold_map" in cols
    rename: Dict[str, str] = {}
    for c in cols:
        if "threshold" not in c:
            continue
        if c == "threshold_map":
            new = "pse"                          # canonical
        elif c == "threshold":
            new = "pse" if not have_map else "pse_prefer"
        else:
            new = c.replace("threshold", "pse")  # keep rest, all lower case
        rename[c] = new

    # avoid accidental collisions
    taken = set(cols)
    final: Dict[str, str] = {}
    for old, new in rename.items():
        if new in taken and new != old:
            k = 1
            cand = f"{new}_renamed"
            while cand in taken:
                k += 1
                cand = f"{new}_renamed{k}"
            new = cand
        taken.add(new)
        final[old] = new
    return final

def rename_threshold_to_pse(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rmap = _build_threshold_rename_map(out.columns)
    if rmap:
        out = out.rename(columns=rmap)
    return out

# ---------- simple derivative ----------
def add_pse_diff(df: pd.DataFrame, pse_col: str = "pse") -> pd.DataFrame:
    out = df.copy()
    pse  = pd.to_numeric(out.get(pse_col), errors="coerce")
    ang  = pd.to_numeric(out.get("standard_angle_abs"), errors="coerce")
    out["pse_diff"] = ang - pse
    return out

# ---------- model predictions ----------
def _within_cue_tables(group: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, float]]:
    """
    For a group (already subset), return a dict:
      cue -> (df_by_angle[['angle','jnd_mean']], jnd_median)
    considering only within-cue rows (standard_cue == comparison_cue).
    """
    g = group.copy()
    g["angle"] = pd.to_numeric(g["standard_angle_abs"], errors="coerce")
    g["jnd"]   = pd.to_numeric(g["jnd"], errors="coerce")
    wc = g[g["standard_cue"].astype(str) == g["comparison_cue"].astype(str)].dropna(subset=["angle", "jnd"])
    out: Dict[str, Tuple[pd.DataFrame, float]] = {}
    if wc.empty:
        return out

    for cue, d in wc.groupby("standard_cue", dropna=False):
        by_ang = (
            d.groupby("angle", dropna=False)["jnd"]
             .mean().reset_index(name="jnd_mean")
             .sort_values("angle", kind="mergesort")
             .reset_index(drop=True)
        )
        jnd_med = float(d["jnd"].median()) if d["jnd"].notna().any() else np.nan
        out[str(cue)] = (by_ang, jnd_med)
    return out

def _lookup_jnd(cue_tables: Dict[str, Tuple[pd.DataFrame, float]], cue: object, angle: float) -> float:
    """
    Get JND for a cue at a given angle:
      - exact angle match if present
      - else nearest angle (by |Δ|)
      - else cue median
      - else NaN
    """
    if cue is None or (isinstance(cue, float) and np.isnan(cue)):
        return np.nan
    key = str(cue)
    if key not in cue_tables:
        return np.nan
    by_ang, jnd_med = cue_tables[key]
    if by_ang.empty or not np.isfinite(angle):
        return jnd_med
    # exact
    m = by_ang["angle"] == angle
    if m.any():
        return float(by_ang.loc[m, "jnd_mean"].iloc[0])
    # nearest
    idx = int((by_ang["angle"] - angle).abs().idxmin())
    return float(by_ang.loc[idx, "jnd_mean"]) if np.isfinite(idx) else jnd_med

def _group_keys_for(df: pd.DataFrame) -> pd.Series:
    """
    Build a grouping key per row:
      - across_frequencies: (dataset, subject_key)
      - else: (dataset, subject_key, standard_center_frequency)
    Returned as a simple string token for easy groupby.
    """
    ds = df["dataset"].astype(str)
    subj = df["subject_key"].astype(str)
    freq = df["standard_center_frequency"]
    # token per policy
    key = np.where(
        ds.eq("across_frequencies"),
        ds + "|" + subj,
        ds + "|" + subj + "|" + freq.astype(str)
    )
    return pd.Series(key, index=df.index, dtype="string")

import numpy as np
import pandas as pd

def add_jnd_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds JND prediction columns per row, using within-cue JNDs from the row's group:
      - jnd_pred_uncertainty = sqrt( (jnd_within[standard_cue]^2 + jnd_within[comparison_cue]^2) / 2 )
      - jnd_pred_scaling     = jnd_within[comparison_cue]
    Group = (dataset, subject_key, standard_center_frequency), except
            dataset=='across_frequencies' where group=(dataset, subject_key).

    Within-cue JND for a cue is selected at the SAME angle if available, else the
    nearest angle, else the cue's median JND within the group.
    """
    out = df.copy()
    # ensure numeric
    out["standard_angle_abs"] = pd.to_numeric(out["standard_angle_abs"], errors="coerce")
    out["jnd"] = pd.to_numeric(out["jnd"], errors="coerce")

    # build group key (uses across_frequencies special-case)
    out["__grp__"] = _group_keys_for(out)

    pred_unc_all = []
    pred_sca_all = []

    for _, g in out.groupby("__grp__", dropna=False):
        cue_tables = _within_cue_tables(g)
        vals_unc, vals_sca = [], []

        for _, r in g.iterrows():
            ang  = float(r["standard_angle_abs"]) if np.isfinite(r["standard_angle_abs"]) else np.nan
            scue = r.get("standard_cue")
            ccue = r.get("comparison_cue")

            jnd_s = _lookup_jnd(cue_tables, scue, ang)
            jnd_c = _lookup_jnd(cue_tables, ccue, ang)

            # uncertainty model: sqrt((jnd_s^2 + jnd_c^2)/2)
            if np.isfinite(jnd_s) and np.isfinite(jnd_c):
                j_unc = float(np.sqrt((jnd_s**2 + jnd_c**2) / 2.0))
            else:
                j_unc = np.nan

            # scaling model: jnd of comparison cue
            j_sca = float(jnd_c) if np.isfinite(jnd_c) else np.nan

            vals_unc.append(j_unc)
            vals_sca.append(j_sca)

        pred_unc_all.append(pd.Series(vals_unc, index=g.index, dtype="float64"))
        pred_sca_all.append(pd.Series(vals_sca, index=g.index, dtype="float64"))

    out["jnd_pred_uncertainty"] = pd.concat(pred_unc_all).sort_index()
    out["jnd_pred_scaling"]     = pd.concat(pred_sca_all).sort_index()
    out = out.drop(columns=["__grp__"])
    return out


def add_prediction_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds error columns = prediction - observed, plus squared versions:
      - pse_pred_error_uncertainty,   pse_pred_error_uncertainty_squared
      - pse_pred_error_scaling,       pse_pred_error_scaling_squared
      - jnd_pred_error_uncertainty,   jnd_pred_error_uncertainty_squared
      - jnd_pred_error_scaling,       jnd_pred_error_scaling_squared
    """
    out = df.copy()

    # numeric safety
    out["pse"] = pd.to_numeric(out.get("pse"), errors="coerce")
    out["jnd"] = pd.to_numeric(out.get("jnd"), errors="coerce")
    for col in ("pse_pred_uncertainty", "pse_pred_scaling",
                "jnd_pred_uncertainty", "jnd_pred_scaling"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # errors
    out["pse_pred_error_uncertainty"] = out.get("pse_pred_uncertainty") - out["pse"]
    out["pse_pred_error_scaling"]     = out.get("pse_pred_scaling")     - out["pse"]
    out["jnd_pred_error_uncertainty"] = out.get("jnd_pred_uncertainty") - out["jnd"]
    out["jnd_pred_error_scaling"]     = out.get("jnd_pred_scaling")     - out["jnd"]

    # squared errors
    out["pse_pred_error_uncertainty_squared"] = out["pse_pred_error_uncertainty"] ** 2
    out["pse_pred_error_scaling_squared"]     = out["pse_pred_error_scaling"] ** 2
    out["jnd_pred_error_uncertainty_squared"] = out["jnd_pred_error_uncertainty"] ** 2
    out["jnd_pred_error_scaling_squared"]     = out["jnd_pred_error_scaling"] ** 2

    return out



def add_pse_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - pse_pred_uncertainty = standard_angle_abs
      - pse_pred_scaling = (JND_within[comparison_cue] / JND_within[standard_cue]) * standard_angle_abs
        within the appropriate group definition.
    """
    out = df.copy()
    out["pse_pred_uncertainty"] = pd.to_numeric(out["standard_angle_abs"], errors="coerce")

    # prepare numeric cols we need
    out["standard_angle_abs"] = pd.to_numeric(out["standard_angle_abs"], errors="coerce")
    out["jnd"] = pd.to_numeric(out["jnd"], errors="coerce")

    # grouping key (handles across_frequencies special case)
    out["__grp__"] = _group_keys_for(out)

    preds = []
    for _, g in out.groupby("__grp__", dropna=False):
        cue_tables = _within_cue_tables(g)
        # row-wise compute scaling pred
        comp = []
        for _, r in g.iterrows():
            ang  = float(r["standard_angle_abs"]) if np.isfinite(r["standard_angle_abs"]) else np.nan
            scue = r.get("standard_cue")
            ccue = r.get("comparison_cue")
            jnd_s = _lookup_jnd(cue_tables, scue, ang)
            jnd_c = _lookup_jnd(cue_tables, ccue, ang)
            val = np.nan
            if np.isfinite(ang) and np.isfinite(jnd_s) and np.isfinite(jnd_c) and (jnd_s != 0):
                val = (jnd_c / jnd_s) * ang
            comp.append(val)
        g_pred = pd.Series(comp, index=g.index, dtype="float64")
        preds.append(g_pred)

    out["pse_pred_scaling"] = pd.concat(preds).sort_index()
    # clean up
    out = out.drop(columns=["__grp__"])
    return out

# ---------- one-shot convenience ----------
def prepare_selected_with_pse_and_predictions(
    path_in: Path | str = DEFAULT_SELECTED,
    path_out: Optional[Path | str] = "Dataframes/psychometrics_params_selected_pse_pred.csv",
) -> pd.DataFrame:
    """
    Load → rename threshold*->pse* (lower case) → add pse_diff
        → add PSE predictions (uncertainty, scaling)
        → add JND predictions (uncertainty, scaling)
        → add prediction error columns for both PSE and JND.
    Save to CSV if path_out is provided.
    """
    df = load_selected_params(path_in)
    df = rename_threshold_to_pse(df)
    df = add_pse_diff(df, pse_col="pse")
    df = add_pse_predictions(df)   # adds: pse_pred_uncertainty, pse_pred_scaling
    df = add_jnd_predictions(df)   # adds: jnd_pred_uncertainty, jnd_pred_scaling
    df = add_prediction_errors(df) # adds: *_pred_error_*

    if path_out is not None:
        outp = Path(path_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
    return df


if __name__ == "__main__":
    df2 = prepare_selected_with_pse_and_predictions()
    print(df2.filter(regex="^(dataset|subject_key|trial_type|standard_center_frequency|standard_angle_abs|pse$|pse_diff|pse_pred_)").head())
