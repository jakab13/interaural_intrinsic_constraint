"""
postprocessing.py

Lightweight utilities for building and reusing the master raw DataFrame,
plus filtering/QA steps for analysis readiness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Tuple

import pandas as pd
import numpy as np

from preprocessing import (
    load_results_tree,
    clean_whitespace,
    summarize_counts,
    SCHEMA_PSYCH_EXAMPLE,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_or_build_raw_master(
    results_root: str | Path,
    df_path: str | Path = "Dataframes/df_raw_master.csv",
    rebuild: bool = False,
    loader_kwargs: Optional[Dict[str, Any]] = None,
    read_kwargs: Optional[Dict[str, Any]] = None,
    save_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load df_raw_master.csv if it exists, otherwise build it from /Results and save.
    """
    results_root = Path(results_root)
    df_path = Path(df_path)

    if loader_kwargs is None:
        loader_kwargs = dict(
            sep="auto",
            clean=True,
            min_rows_per_file=48,
            short_file="skip",
            drop_columns=("inter_stimulus_interval", "mixing_gain"),
        )
    if read_kwargs is None:
        read_kwargs = dict(low_memory=False)
    if save_kwargs is None:
        save_kwargs = dict(index=False)

    if df_path.exists() and not rebuild:
        logger.info(f"Loading existing master DataFrame: {df_path}")
        df = pd.read_csv(df_path, **read_kwargs)
        df = clean_whitespace(df)
        return df

    logger.info("Building master DataFrame from raw /Results ...")
    df = load_results_tree(results_root, **loader_kwargs)

    df_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving master DataFrame to: {df_path}")
    df.to_csv(df_path, **save_kwargs)

    return df


# -----------------------------------------------------------------------------
# Filtering & sanity checks
# -----------------------------------------------------------------------------

def _resolve_col_generic(df: pd.DataFrame, name: str, aliases: Sequence[str]) -> Optional[str]:
    """Return the first column present among [name] + aliases; None if none exist."""
    candidates = [name] + list(aliases or [])
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_standard_angle_abs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'standard_angle_abs' column exists. If missing, derive from a plausible angle column.
    """
    if "standard_angle_abs" in df.columns:
        return df

    angle_candidates = [
        "standard_angle", "std_angle", "standard", "standard_angle_deg",
        "standard_angle_degree", "standard_degrees"
    ]
    col = _resolve_col_generic(df, "standard_angle", angle_candidates[1:])
    if col is None:
        return df

    ang = pd.to_numeric(df[col], errors="coerce")
    df = df.copy()
    df["standard_angle_abs"] = ang.abs()
    return df


def restrict_combined_cue_frequencies(
    df: pd.DataFrame,
    allowed: Sequence[float] = (1300, 500),
    dataset_col: str = "dataset",
    freq_col: str = "standard_center_frequency",
    freq_aliases: Sequence[str] = (
        "standard_freq", "std_freq", "standard_frequency", "std_frequency",
        "center_frequency", "frequency_hz", "frequency"
    ),
) -> pd.DataFrame:
    """
    Keep all rows for datasets != 'combined_cue'.
    For 'combined_cue', retain only rows whose standard_center_frequency is in `allowed`.
    """
    if dataset_col not in df.columns:
        raise ValueError(f"Column '{dataset_col}' not found. Available: {list(df.columns)}")

    fcol = _resolve_col_generic(df, freq_col, freq_aliases)
    if fcol is None:
        raise ValueError(
            f"Frequency column not found. Looked for '{freq_col}' and aliases {list(freq_aliases)}. "
            f"Available: {list(df.columns)}"
        )

    out = df.copy()
    freq_num = pd.to_numeric(out[fcol], errors="coerce")
    allowed_set = set(float(x) for x in allowed)

    mask = out[dataset_col] != "combined_cue"
    mask |= ((out[dataset_col] == "combined_cue") & (freq_num.isin(list(allowed_set))))

    filtered = out.loc[mask].reset_index(drop=True)
    removed = len(out) - len(filtered)
    if removed > 0:
        logger.info(f"restrict_combined_cue_frequencies: removed {removed} row(s) outside allowed {sorted(allowed_set)} for dataset='combined_cue'.")
    return filtered


def sanity_check_min_counts(
    df: pd.DataFrame,
    min_n: int = 192,
    groupby: Sequence[str] = ("dataset", "subject", "trial_type", "standard_center_frequency", "standard_angle_abs"),
    alias_map: Optional[Dict[str, Sequence[str]]] = None,
    raise_on_fail: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize counts by the given grouping and ensure none are below `min_n`.
    Returns (counts_tidy, offenders_df). Optionally raises if offenders exist.
    """
    df = _ensure_standard_angle_abs(df)

    default_alias = {
        "subject": ["subject_key", "subject_label", "subject_id"],
        "trial_type": ["condition", "trialtype", "trial_type_name"],
        "standard_center_frequency": [
            "standard_freq", "std_freq", "standard_frequency", "std_frequency",
            "center_frequency", "frequency_hz", "frequency"
        ],
        "standard_angle_abs": [],
    }
    if alias_map is None:
        alias_map = default_alias
    else:
        merged = default_alias.copy()
        merged.update(alias_map)
        alias_map = merged

    counts_tidy, _ = summarize_counts(
        df,
        groupby=list(groupby),
        alias_map=alias_map,
        require=list(groupby),
        wide=False,
        sort=True,
        dropna=False,
        count_name="n",
    )

    offenders = counts_tidy[counts_tidy["n"] < int(min_n)].copy()
    if len(offenders) and raise_on_fail:
        preview = offenders.head(10).to_string(index=False)
        raise AssertionError(
            f"Sanity check failed: {len(offenders)} group(s) have counts < {min_n}.\n"
            f"First offenders:\n{preview}\n"
            f"(Pass raise_on_fail=False to get the offenders DataFrame without raising.)"
        )

    return counts_tidy, offenders


# --- Min-count filtering (preserve original columns) -------------------------

def _default_alias_map_for_groups() -> Dict[str, Sequence[str]]:
    return {
        "subject": ["subject_key", "subject_label", "subject_id"],
        "trial_type": ["condition", "trialtype", "trial_type_name"],
        "standard_center_frequency": [
            "standard_freq", "std_freq", "standard_frequency", "std_frequency",
            "center_frequency", "frequency_hz", "frequency"
        ],
        "standard_angle_abs": [],
        "dataset": [],
    }


def _resolve_groupby_mapping(
    df: pd.DataFrame,
    groupby: Sequence[str],
    alias_map: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, str]:
    df = _ensure_standard_angle_abs(df)
    _alias = _default_alias_map_for_groups()
    if alias_map:
        merged = _alias.copy()
        merged.update(alias_map)
        _alias = merged

    mapping: Dict[str, str] = {}
    for name in groupby:
        actual = _resolve_col_generic(df, name, _alias.get(name, []))
        if actual is None:
            raise ValueError(f"Could not resolve grouping column '{name}' in DataFrame. "
                             f"Tried aliases: {_alias.get(name, [])}. "
                             f"Available: {list(df.columns)}")
        mapping[name] = actual
    return mapping


def filter_min_count_groups(
    df: pd.DataFrame,
    min_n: int = 192,
    groupby: Sequence[str] = ("dataset", "subject", "trial_type", "standard_center_frequency", "standard_angle_abs"),
    alias_map: Optional[Dict[str, Sequence[str]]] = None,
    return_reports: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Remove all rows belonging to groups whose count is < min_n, while preserving
    the original DataFrame's columns (no merges, no suffixes).
    """
    df = _ensure_standard_angle_abs(df)

    mapping = _resolve_groupby_mapping(df, groupby, alias_map)
    left_keys = [mapping[g] for g in groupby]

    counts_tidy, _ = summarize_counts(
        df,
        groupby=list(groupby),
        alias_map=alias_map or _default_alias_map_for_groups(),
        require=list(groupby),
        wide=False,
        sort=False,
        dropna=False,
        count_name="n",
    )
    offenders = counts_tidy[counts_tidy["n"] < int(min_n)].copy()

    n_per_row = df.groupby(left_keys, dropna=False)[left_keys[0]].transform("size")
    mask = n_per_row >= int(min_n)
    df_filtered = df.loc[mask].copy()

    removed_rows = int((~mask).sum())
    if removed_rows > 0:
        logger.info(
            f"filter_min_count_groups: removed {removed_rows} row(s) from groups with n < {min_n}. "
            f"Remaining rows: {len(df_filtered)}"
        )

    if return_reports:
        return df_filtered, counts_tidy, offenders
    else:
        return df_filtered, None, None


# -----------------------------------------------------------------------------
# Additional QC helpers and improved pipeline reporting
# -----------------------------------------------------------------------------

def per_file_row_counts(
    df: pd.DataFrame,
    source_col: str = "source_name",
) -> pd.DataFrame:
    """
    Return a tidy table of per-file row counts using the given source column.
    """
    if source_col not in df.columns:
        raise ValueError(f"per_file_row_counts: '{source_col}' column not found. Available: {list(df.columns)}")
    return (
        df.groupby(source_col, dropna=False)
          .size()
          .reset_index(name="n_rows")
          .sort_values("n_rows")
          .reset_index(drop=True)
    )


def drop_short_files(
    df: pd.DataFrame,
    min_rows_per_file: int = 48,
    source_col: str = "source_name",
    return_report: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Remove *entire files* whose total number of rows is < min_rows_per_file.
    This is useful if df_raw_master.csv was built before QC was introduced.
    """
    counts = per_file_row_counts(df, source_col=source_col)
    offenders = counts[counts["n_rows"] < int(min_rows_per_file)]
    if not offenders.empty:
        bad = set(offenders[source_col].tolist())
        out = df[~df[source_col].isin(bad)].copy()
        logger.info(f"drop_short_files: removed {len(bad)} file(s) and {len(df) - len(out)} row(s) below {min_rows_per_file} rows.")
    else:
        out = df.copy()
    if return_report:
        return out, offenders
    return out, None


def apply_core_filters(
    df: pd.DataFrame,
    allowed_freqs: Sequence[float] = (1300, 500),
    min_n_per_group: int = 192,
    groupby: Sequence[str] = ("dataset", "subject", "trial_type", "standard_center_frequency", "standard_angle_abs"),
    alias_map: Optional[Dict[str, Sequence[str]]] = None,
    min_rows_per_file: int = 48,
    source_col: str = "source_name",
    return_reports: bool = True,
    add_control_flag: bool = True,          # <--- add this
    angle_tolerance: float = 1e-3,          # <--- and this
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience pipeline:
      0) Drop short files
      1) Restrict 'combined_cue' freqs
      2) Drop groups with n < min_n_per_group
      3) (optional) Add 'is_control' for single_cue
    """
    reports: Dict[str, pd.DataFrame] = {}

    file_counts_before = per_file_row_counts(df, source_col=source_col)
    reports["file_counts_before"] = file_counts_before
    df0, short_files = drop_short_files(df, min_rows_per_file=min_rows_per_file, source_col=source_col, return_report=True)
    reports["short_files_removed"] = short_files

    df1 = restrict_combined_cue_frequencies(df0, allowed=allowed_freqs)

    df2, counts_pre, offenders = filter_min_count_groups(
        df1,
        min_n=min_n_per_group,
        groupby=groupby,
        alias_map=alias_map,
        return_reports=True,
    )

    counts_post, _ = summarize_counts(
        df2,
        groupby=list(groupby),
        alias_map=alias_map or _default_alias_map_for_groups(),
        require=list(groupby),
        wide=False,
        sort=True,
        dropna=False,
        count_name="n",
    )

    reports["counts_pre"] = counts_pre
    reports["offenders"] = offenders
    reports["counts_post"] = counts_post

    # --- New: control flag annotation (post-filter) ---
    # inside apply_core_filters(...) just before returning
    if add_control_flag:
        df2 = add_is_control(
            df2,
            angle_tolerance=angle_tolerance,
            trial_type_col="trial_type",
            trial_type_aliases=("condition", "trialtype", "trial_type_name"),
            datasets=("single_cue", "across_frequencies"),
        )

    return df2, reports


__all__ = [
    "get_or_build_raw_master",
    "restrict_combined_cue_frequencies",
    "sanity_check_min_counts",
    "filter_min_count_groups",
    "per_file_row_counts",
    "drop_short_files",
    "apply_core_filters",
    "summarize_counts",
    "SCHEMA_PSYCH_EXAMPLE",
]


def add_is_control_single_cue(
    df: pd.DataFrame,
    angle_tolerance: float = 1e-3,
    trial_type_col: str = "trial_type",
    trial_type_aliases: Sequence[str] = ("condition", "trialtype", "trial_type_name"),
) -> pd.DataFrame:
    """
    Add a boolean column 'is_control' for the single_cue dataset:
      - For standard_angle_abs ≈ 5 or 8 or 10, control is trial_type == "ILD-->ILD".
      - For non-integer standard_angle_abs, control is trial_type == "ITD-->ITD".
    All other rows are False. Keeps existing columns; derives standard_angle_abs if needed.

    Parameters
    ----------
    angle_tolerance : float
        Numeric tolerance for integer comparisons (e.g., 5.000 vs 5).
    """
    # Ensure |angle| column exists and numeric
    df = _ensure_standard_angle_abs(df).copy()
    ang = pd.to_numeric(df["standard_angle_abs"], errors="coerce")

    # Resolve trial_type column (alias-aware)
    tt_col = _resolve_col_generic(df, trial_type_col, trial_type_aliases)
    if tt_col is None:
        raise ValueError(
            f"add_is_control_single_cue: could not find trial type column. "
            f"Tried '{trial_type_col}' and aliases {list(trial_type_aliases)}. "
            f"Available: {list(df.columns)}"
        )

    # Normalize trial_type strings for robust matching
    tt_norm = (
        df[tt_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)  # remove spaces
        .str.upper()
    )

    # Identify integer-like angles, and specifically those near 5 or 8
    is_int_like = (ang - ang.round()).abs() <= angle_tolerance
    is_5 = (ang - 5).abs() <= angle_tolerance
    is_8 = (ang - 8).abs() <= angle_tolerance
    is_10 = (ang - 10).abs() <= angle_tolerance
    is_integer_5_or_8_or_10 = is_int_like & (is_5 | is_8 | is_10)

    # Non-integer group (anything not integer-like)
    is_non_integer = ~is_int_like

    # Only apply to single_cue dataset
    is_single_cue = df["dataset"].astype(str).str.strip().str.lower().eq("single_cue")

    # Control definitions
    ctrl_integer = is_integer_5_or_8_or_10 & tt_norm.eq("ILD-->ILD")
    ctrl_nonint = is_non_integer & tt_norm.eq("ITD-->ITD")

    df["is_control"] = (is_single_cue & (ctrl_integer | ctrl_nonint)).astype("boolean")
    return df


__all__ += ["add_is_control_single_cue"]


def add_is_control_across_frequencies(
    df: pd.DataFrame,
    angle_tolerance: float = 1e-3,
    trial_type_col: str = "trial_type",
    trial_type_aliases: Sequence[str] = ("condition", "trialtype", "trial_type_name"),
) -> pd.DataFrame:
    """
    Add/extend boolean 'is_control' for the across_frequencies dataset:
      - For standard_angle_abs ≈ 5 or 8, control is trial_type == "ILD-->ILD".
      - For non-integer standard_angle_abs, control is trial_type == "ITD-->ITD".
    All other rows are False for this dataset. Keeps existing columns; derives
    standard_angle_abs if needed. If 'is_control' already exists, this ORs into it.
    """
    # Ensure |angle| column exists and numeric
    df = _ensure_standard_angle_abs(df).copy()
    ang = pd.to_numeric(df["standard_angle_abs"], errors="coerce")

    # Resolve trial_type column (alias-aware)
    tt_col = _resolve_col_generic(df, trial_type_col, trial_type_aliases)
    if tt_col is None:
        raise ValueError(
            f"add_is_control_across_frequencies: could not find trial type column. "
            f"Tried '{trial_type_col}' and aliases {list(trial_type_aliases)}. "
            f"Available: {list(df.columns)}"
        )

    # Normalize trial_type strings for robust matching
    tt_norm = (
        df[tt_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)  # remove spaces
        .str.upper()
    )

    # Integer-like vs non-integer angles
    is_int_like = (ang - ang.round()).abs() <= angle_tolerance
    is_5 = (ang - 5).abs() <= angle_tolerance
    is_8 = (ang - 8).abs() <= angle_tolerance
    is_integer_5_or_8 = is_int_like & (is_5 | is_8)
    is_non_integer = ~is_int_like

    # Only apply to across_frequencies
    is_across = df["dataset"].astype(str).str.strip().str.lower().eq("across_frequencies")

    # Control definitions
    ctrl_integer = is_integer_5_or_8 & tt_norm.eq("ILD-->ILD")
    ctrl_nonint = is_non_integer & tt_norm.eq("ITD-->ITD")
    new_flag = (is_across & (ctrl_integer | ctrl_nonint))

    # If is_control already exists, OR into it; otherwise create
    if "is_control" in df.columns:
        df["is_control"] = (df["is_control"].astype("boolean").fillna(False) | new_flag).astype("boolean")
    else:
        df["is_control"] = new_flag.astype("boolean")
    return df


def add_is_control(
    df: pd.DataFrame,
    angle_tolerance: float = 1e-3,
    trial_type_col: str = "trial_type",
    trial_type_aliases: Sequence[str] = ("condition", "trialtype", "trial_type_name"),
    datasets: Sequence[str] = ("single_cue", "across_frequencies"),
) -> pd.DataFrame:
    """
    Apply control flags across requested datasets. Preserves existing 'is_control'
    by OR-ing dataset-specific flags.
    """
    out = df
    if any(d.lower() == "single_cue" for d in datasets) and "add_is_control_single_cue" in globals():
        out = add_is_control_single_cue(
            out,
            angle_tolerance=angle_tolerance,
            trial_type_col=trial_type_col,
            trial_type_aliases=trial_type_aliases,
        )
    if any(d.lower() == "across_frequencies" for d in datasets):
        out = add_is_control_across_frequencies(
            out,
            angle_tolerance=angle_tolerance,
            trial_type_col=trial_type_col,
            trial_type_aliases=trial_type_aliases,
        )
    return out


__all__ += ["add_is_control_across_frequencies", "add_is_control"]

