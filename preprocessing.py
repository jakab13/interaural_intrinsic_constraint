"""
preprocessing.py

Utilities to load and preprocess text-based tabular data for the
Interaural Intrinsic Constraint project.

Integrated features
-------------------
- Robust single-file and multi-file loaders for .txt (CSV/TSV) with auto delimiter detection.
- Whitespace cleanup for headers and string cells (handles tabs and BOMs).
- Project-aware loaders for the canonical /Results tree with disambiguating subject keys.
- Optional schema validation with aliases and dtype coercion.
- Flexible summarizer to count rows grouped by arbitrary columns (alias-aware).
- Quality control: drop redundant columns and require a minimum number of rows per file.
"""

from __future__ import annotations

import re
import logging
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable, Iterable, Optional, Union, List, Sequence, Dict, Literal, Tuple
)

import pandas as pd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # loaders
    "load_txt_to_df",
    "load_txts_from_folder",
    "scan_results_tree",
    "load_results_tree",
    # utilities
    "default_subject_parser",
    "clean_whitespace",
    "summarize_counts",
    # schema
    "SchemaSpec",
    "validate_schema",
    "SCHEMA_PSYCH_EXAMPLE",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p


def default_subject_parser(p: Union[str, Path]) -> str:
    """
    Heuristic subject ID parser from file path / name.
    Falls back gracefully if common patterns not found.
    """
    stem = Path(p).stem
    patterns = [
        r"subject[_-]?([A-Za-z0-9]+)",
        r"subj[_-]?([A-Za-z0-9]+)",
        r"s[_-]?([A-Za-z0-9]+)",
        r"participant[_-]?([A-Za-z0-9]+)",
        r"[^\w]?S(?:ubject)?[_-]?(\d+)",
        r"vp[_-]?(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _normalize_dataset_name(name: str) -> str:
    name = name.strip().lower().replace("-", "_")
    mapping = {
        "single_cue": "single_cue",
        "combined_cue": "combined_cue",
        "across_frequencies": "across_frequencies",
    }
    return mapping.get(name, name)


def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with BOMs removed and whitespace (incl. tabs) stripped
    from headers and object columns.
    """
    out = df.copy()
    out.columns = (
        pd.Index(out.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
    )
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        out[c] = out[c].astype("string").str.strip()
    return out


def _read_csv_auto(
    filepath: Union[str, Path],
    dtype: Optional[dict] = None,
    na_values: Optional[Iterable] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Robust CSV/TSV reader with automatic delimiter detection and fallbacks.
    """
    # 1) Try pandas' built-in automatic detection (python engine)
    try:
        df = pd.read_csv(
            filepath,
            sep=None,
            engine="python",
            dtype=dtype,
            na_values=na_values,
            encoding=encoding,
            skipinitialspace=True,
        )
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 2) Try tab
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype=dtype,
            na_values=na_values,
            encoding=encoding,
        )
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 3) Try comma
    try:
        df = pd.read_csv(
            filepath,
            sep=",",
            dtype=dtype,
            na_values=na_values,
            encoding=encoding,
        )
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 4) Last resort: regex split on either tabs or commas
    df = pd.read_csv(
        filepath,
        sep=r"[\t,]+",
        engine="python",
        dtype=dtype,
        na_values=na_values,
        encoding=encoding,
    )
    return df


def _drop_columns_safe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    if not cols:
        return df
    to_drop = [c for c in cols if c in df.columns]
    if to_drop:
        return df.drop(columns=to_drop)
    return df


def _check_min_rows(name: str, df: pd.DataFrame, min_rows: int, behavior: str = "skip") -> bool:
    """
    Return True if df passes the min_rows check. Otherwise:
      - 'skip': log warning and return False (caller should skip the file)
      - 'warn': log warning and return True (keep the file)
      - 'raise': raise ValueError
    """
    if min_rows is None or min_rows <= 0:
        return True
    if len(df) >= min_rows:
        return True
    msg = f"File '{name}' has {len(df)} rows (< {min_rows}); "
    if behavior == "skip":
        logger.warning(msg + "skipping.")
        return False
    elif behavior == "warn":
        logger.warning(msg + "keeping with warning.")
        return True
    elif behavior == "raise":
        raise ValueError(msg + "aborting.")
    else:
        logger.warning(msg + f"unknown behavior '{behavior}', defaulting to 'skip'.")
        return False


# -----------------------------------------------------------------------------
# Generic loaders
# -----------------------------------------------------------------------------

def load_txt_to_df(
    filepath: Union[str, Path],
    sep: Union[str, None] = "auto",
    encoding: str = "utf-8",
    dtype: Optional[dict] = None,
    na_values: Optional[Iterable] = None,
    clean: bool = True,
    drop_columns: Optional[Iterable[str]] = ("inter_stimulus_interval", "mixing_gain"),
    min_rows: Optional[int] = None,
    on_short: str = "skip",  # 'skip'|'warn'|'raise' (only used if min_rows is not None)
) -> pd.DataFrame:
    """
    Load a .txt file containing delimiter-separated values into a pandas DataFrame.
    """
    filepath = _ensure_path(filepath)
    if sep in ("auto", None):
        df = _read_csv_auto(filepath, dtype=dtype, na_values=na_values, encoding=encoding)
    else:
        df = pd.read_csv(filepath, sep=sep, encoding=encoding, dtype=dtype, na_values=na_values)

    if clean:
        df = clean_whitespace(df)

    # drop redundant columns
    if drop_columns:
        df = _drop_columns_safe(df, drop_columns)

    # min-rows QC (single-file path)
    if min_rows is not None:
        ok = _check_min_rows(str(filepath), df, int(min_rows), behavior=on_short)
        if not ok:
            # Return an empty DataFrame with a helpful attribute
            df = df.iloc[0:0].copy()
            df.attrs["skipped_due_to_short_length"] = True

    return df


def load_txts_from_folder(
    folder: Union[str, Path],
    pattern: str = "*.txt",
    sep: Union[str, None] = "auto",
    encoding: str = "utf-8",
    add_source: bool = True,
    subject_parser: Optional[Callable[[Union[str, Path]], str]] = default_subject_parser,
    dtype: Optional[dict] = None,
    na_values: Optional[Iterable] = None,
    recursive: bool = False,
    sort_files: bool = True,
    clean: bool = True,
    drop_columns: Optional[Iterable[str]] = ("inter_stimulus_interval", "mixing_gain"),
    min_rows_per_file: Optional[int] = None,
    short_file: Literal["skip", "warn", "raise"] = "skip",
) -> pd.DataFrame:
    """
    Load all .txt files in a folder, concatenate into a single DataFrame, and add metadata.
    """
    folder = _ensure_path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files: List[Path] = list(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {folder}")

    if sort_files:
        files = sorted(files)

    frames = []
    for f in files:
        try:
            if sep in ("auto", None):
                df = _read_csv_auto(f, dtype=dtype, na_values=na_values, encoding=encoding)
            else:
                df = pd.read_csv(f, sep=sep, encoding=encoding, dtype=dtype, na_values=na_values)

            if clean:
                df = clean_whitespace(df)

            if drop_columns:
                df = _drop_columns_safe(df, drop_columns)

            if min_rows_per_file is not None and not _check_min_rows(f.name, df, int(min_rows_per_file), behavior=short_file):
                continue

            if add_source:
                df["source_path"] = str(f.resolve())
                df["source_name"] = f.name
            if subject_parser is not None:
                try:
                    df["subject_id"] = subject_parser(f)
                except Exception as e:
                    logger.warning(f"subject_parser failed for {f}: {e}")
                    df["subject_id"] = None
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            if short_file == "skip":
                logger.warning(f"Skipping file due to error: {f}")
                continue
            else:
                raise

    if not frames:
        raise FileNotFoundError("No valid files remained after QC.")

    out = pd.concat(frames, ignore_index=True)

    front_cols = []
    if add_source:
        front_cols += ["source_name", "source_path"]
    if subject_parser is not None:
        front_cols += ["subject_id"]
    front_cols = [c for c in front_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in front_cols]
    out = out.loc[:, front_cols + other_cols]

    logger.info(f"Loaded {len(files)} file(s); concatenated rows: {len(out)}")
    return out


# -----------------------------------------------------------------------------
# Project-specific scanners/loaders for /Results
# -----------------------------------------------------------------------------

def scan_results_tree(
    root: Union[str, Path],
    datasets: Sequence[str] = ("single_cue", "combined_cue", "across_frequencies"),
    pattern: str = "*.txt",
    recursive_subject_subdirs: bool = True,
) -> pd.DataFrame:
    """
    Scan the canonical /Results tree for .txt files and return a manifest.
    """
    root = _ensure_path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    datasets_norm = {_normalize_dataset_name(d) for d in datasets}
    rows: List[Dict[str, str]] = []

    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = _normalize_dataset_name(dataset_dir.name)
        if dataset not in datasets_norm:
            continue

        subj_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
        for sdir in subj_dirs:
            if recursive_subject_subdirs:
                files = list(sdir.rglob(pattern))
            else:
                files = list(sdir.glob(pattern))

            subject_label = sdir.name.lower()
            m = re.search(r"(vp)[_-]?(\d+)", subject_label, flags=re.IGNORECASE)
            if m:
                subject_label = f"{m.group(1).lower()}_{m.group(2)}"

            for f in files:
                rows.append(
                    {
                        "dataset": dataset,
                        "subject_label": subject_label,
                        "subject_key": f"{dataset}:{subject_label}",
                        "source_path": str(Path(f).resolve()),
                        "source_name": Path(f).name,
                    }
                )

    if not rows:
        raise FileNotFoundError(
            f"No files found under {root} matching {pattern} in datasets {sorted(datasets_norm)}"
        )

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(["dataset", "subject_label", "source_name"]).reset_index(drop=True)
    return manifest


# ---- Schema validation -------------------------------------------------------

@dataclass
class SchemaSpec:
    """
    A flexible schema specification for validating and coercing tabular data.

    Attributes
    ----------
    name : str
        Human-readable name of the schema.
    required : Dict[str, str]
        Mapping of *canonical* column names -> dtype string (e.g., 'int', 'float', 'string', 'bool').
    optional : Dict[str, str]
        Mapping of optional columns (coerced if present).
    aliases : Dict[str, str]
        Mapping of alternative header names -> canonical names (case-insensitive matching is applied).
    allow_extra : bool
        If False, error when extra/unexpected columns are present.
    """
    name: str
    required: Dict[str, str]
    optional: Dict[str, str] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    allow_extra: bool = True


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _apply_aliases(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    if not aliases:
        return df
    ali = {k.strip().lower(): v for k, v in aliases.items()}  # case-insensitive map
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in ali:
            rename_map[col] = ali[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    TRUE = {"1", "true", "t", "yes", "y", "correct", "hit"}
    FALSE = {"0", "false", "f", "no", "n", "incorrect", "miss"}
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out = out.mask(s.isin(TRUE), True)
    out = out.mask(s.isin(FALSE), False)
    # numeric fallback
    try:
        num = pd.to_numeric(series, errors="coerce")
        out = out.fillna(num == 1)
        out = out.mask(num == 0, False)
    except Exception:
        pass
    return out


def _coerce_dtype(series: pd.Series, dtype_str: str) -> pd.Series:
    dt = dtype_str.strip().lower()
    if dt in {"int", "int64", "integer"}:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if dt in {"float", "float64", "double"}:
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if dt in {"string", "str", "text"}:
        return series.astype("string")
    if dt in {"bool", "boolean"}:
        return _coerce_bool(series)
    if dt in {"category", "categorical"}:
        return series.astype("category")
    try:
        return series.astype(dtype_str)
    except Exception:
        return series


def validate_schema(
    df: pd.DataFrame,
    schema: SchemaSpec,
    coerce: bool = True,
    rename_aliases: bool = True,
    errors: Literal["raise", "warn"] = "raise",
) -> pd.DataFrame:
    """
    Validate presence of required columns (with aliases), coerce dtypes, and optionally
    warn/raise on unexpected columns. Returns a *new* DataFrame.
    """
    original_cols = list(df.columns)
    df2 = clean_whitespace(_normalize_headers(df))

    if rename_aliases:
        df2 = _apply_aliases(df2, schema.aliases)

    # Check required columns
    missing = [c for c in schema.required if c not in df2.columns]
    if missing:
        suggestions = {m: difflib.get_close_matches(m, df2.columns, n=2) for m in missing}
        msg = (
            f"Missing required columns for schema '{schema.name}': {missing}. "
            f"Closest matches: {suggestions}. Original headers: {original_cols}"
        )
        if errors == "raise":
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # Coerce dtypes
    if coerce:
        for col, typ in {**schema.optional, **schema.required}.items():
            if col in df2.columns:
                try:
                    df2[col] = _coerce_dtype(df2[col], typ)
                except Exception as e:
                    m = f"Failed to coerce column '{col}' to {typ}: {e}"
                    if errors == "raise":
                        raise TypeError(m)
                    else:
                        logger.warning(m)

    # Unexpected columns
    if not schema.allow_extra:
        unexpected = [c for c in df2.columns if c not in schema.required and c not in schema.optional]
        if unexpected:
            msg = f"Unexpected columns under schema '{schema.name}': {unexpected}"
            if errors == "raise":
                raise ValueError(msg)
            else:
                logger.warning(msg)

    # Order columns: required, optional, then the rest
    front = [c for c in list(schema.required) + list(schema.optional) if c in df2.columns]
    rest = [c for c in df2.columns if c not in front]
    df2 = df2.loc[:, front + rest]
    return df2


# Starter schema — adjust to your actual headers as needed
SCHEMA_PSYCH_EXAMPLE = SchemaSpec(
    name="psychophysics_base",
    required={
        "trial": "int",
        "standard_angle": "float",
        "comparison_angle": "float",
        "response": "string",
    },
    optional={
        "correct": "bool",
        "rt": "float",
        "cue": "string",
        "block": "string",
        "condition": "string",
        "trial_type": "string",
        "standard_center_frequency": "float",
    },
    aliases={
        # trial
        "trialindex": "trial",
        "trial_index": "trial",
        "trialnum": "trial",
        "trialnumber": "trial",
        "trial nr": "trial",
        # angles
        "std_angle": "standard_angle",
        "standard": "standard_angle",
        "standardangle": "standard_angle",
        "standard angle": "standard_angle",
        "comparison": "comparison_angle",
        "cmp_angle": "comparison_angle",
        "comparisonangle": "comparison_angle",
        "comparison angle": "comparison_angle",
        # response
        "resp": "response",
        "button": "response",
        "key": "response",
        # misc
        "rt_ms": "rt",
        "reaction_time": "rt",
        "cue_type": "cue",
        "trial_type_name": "trial_type",
        # frequency
        "standard_freq": "standard_center_frequency",
        "std_freq": "standard_center_frequency",
        "standard_frequency": "standard_center_frequency",
        "std_frequency": "standard_center_frequency",
        "center_frequency": "standard_center_frequency",
        "frequency_hz": "standard_center_frequency",
        "frequency": "standard_center_frequency",
    },
    allow_extra=True,
)


def load_results_tree(
    root: Union[str, Path],
    datasets: Sequence[str] = ("single_cue", "combined_cue", "across_frequencies"),
    pattern: str = "*.txt",
    sep: Union[str, None] = "auto",
    encoding: str = "utf-8",
    dtype: Optional[dict] = None,
    na_values: Optional[Iterable] = None,
    recursive_subject_subdirs: bool = True,
    add_source_columns: bool = True,
    clean: bool = True,
    schema: Optional[SchemaSpec] = None,
    schema_errors: Literal["raise", "warn", "skip_file"] = "raise",
    drop_columns: Optional[Iterable] = ("inter_stimulus_interval", "mixing_gain"),
    min_rows_per_file: int = 12,
    short_file: Literal["skip", "warn", "raise"] = "skip",
) -> pd.DataFrame:
    """
    Load and concatenate all .txt files found in the /Results tree, with optional schema validation,
    redundant column dropping, and per-file minimum-length QC.
    """
    manifest = scan_results_tree(
        root=root,
        datasets=datasets,
        pattern=pattern,
        recursive_subject_subdirs=recursive_subject_subdirs,
    )

    frames: List[pd.DataFrame] = []
    kept_files = 0
    for row in manifest.itertuples(index=False):
        f = Path(row.source_path)
        try:
            if sep in ("auto", None):
                df = _read_csv_auto(f, dtype=dtype, na_values=na_values, encoding=encoding)
            else:
                df = pd.read_csv(f, sep=sep, encoding=encoding, dtype=dtype, na_values=na_values)

            if clean:
                df = clean_whitespace(df)

            # drop redundant columns early
            if drop_columns:
                df = _drop_columns_safe(df, drop_columns)

            # schema validation per file
            if schema is not None:
                try:
                    df = validate_schema(df, schema=schema, coerce=True, rename_aliases=True, errors="raise")
                except Exception as e:
                    msg = f"[{row.dataset} / {row.subject_label} / {row.source_name}] schema validation failed: {e}"
                    if schema_errors == "raise":
                        raise ValueError(msg)
                    elif schema_errors == "warn":
                        logger.warning(msg)
                    elif schema_errors == "skip_file":
                        logger.warning(msg + " -- skipping file")
                        continue

            # min-rows QC
            if not _check_min_rows(row.source_name, df, int(min_rows_per_file), behavior=short_file):
                continue  # skip this file

            # attach metadata (only after QC)
            df.insert(0, "dataset", row.dataset)
            df.insert(1, "subject_label", row.subject_label)
            df.insert(2, "subject_key", row.subject_key)
            if add_source_columns:
                df["source_name"] = row.source_name
                df["source_path"] = row.source_path

            frames.append(df)
            kept_files += 1

        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            if schema_errors == "skip_file" or short_file == "skip":
                logger.warning(f"Skipping file due to error: {f}")
                continue
            else:
                raise

    if not frames:
        raise FileNotFoundError("No valid files remained after QC/validation.")

    out = pd.concat(frames, ignore_index=True)

    front_cols = ["dataset", "subject_label", "subject_key"]
    if add_source_columns:
        front_cols += ["source_name", "source_path"]
    front_cols = [c for c in front_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in front_cols]
    out = out.loc[:, front_cols + other_cols]

    logger.info(
        f"Loaded {kept_files:,} file(s) after QC from {out['dataset'].nunique()} dataset(s); "
        f"subjects (unique subject_key): {out['subject_key'].nunique()}; "
        f"total rows: {len(out):,}"
    )
    return out


# -----------------------------------------------------------------------------
# Flexible summarizer (alias-aware)
# -----------------------------------------------------------------------------

def summarize_counts(
    df: pd.DataFrame,
    groupby: Optional[Sequence[str]] = None,
    alias_map: Optional[Dict[str, Sequence[str]]] = None,
    require: Optional[Sequence[str]] = None,
    sort: bool = True,
    dropna: bool = False,
    count_name: str = "n_trials",
    wide: bool = False,
    pivot_index: Optional[Sequence[str]] = None,
    pivot_columns: Optional[Sequence[str]] = None,
    fill_value: int = 0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Flexible trial-count summarizer with optional alias resolution and pivoting.
    """
    # Defaults (backward compatible)
    default_groupby = ["dataset", "subject_key", "trial_type", "standard_center_frequency"]
    if groupby is None:
        groupby = default_groupby

    # Default alias map for common columns
    default_alias_map: Dict[str, Sequence[str]] = {
        "trial_type": ["condition", "trialtype", "trial_type_name"],
        "standard_center_frequency": [
            "standard_freq", "std_freq", "standard_frequency", "std_frequency",
            "center_frequency", "frequency_hz", "frequency"
        ],
    }
    if alias_map is None:
        alias_map = default_alias_map
    else:
        merged = default_alias_map.copy()
        merged.update(alias_map)
        alias_map = merged

    if require is None:
        require = []

    # Resolve each groupby name to an actual column in df
    resolved: List[str] = []
    missing_required: List[str] = []
    resolution_map: Dict[str, Optional[str]] = {}

    for name in groupby:
        actual: Optional[str] = None
        if name in df.columns:
            actual = name
        else:
            for candidate in alias_map.get(name, []):
                if candidate in df.columns:
                    actual = candidate
                    break

        if actual is None:
            resolution_map[name] = None
            if name in require:
                missing_required.append(name)
            else:
                logger.warning(f"summarize_counts: dropping missing group column '{name}' "
                               f"(aliases tried: {alias_map.get(name, [])})")
        else:
            resolution_map[name] = actual
            resolved.append(actual)

    if missing_required:
        raise ValueError(f"summarize_counts: required columns missing after alias resolution: {missing_required}")

    if not resolved:
        raise ValueError("summarize_counts: no valid grouping columns found. "
                         f"Requested={groupby}, available={list(df.columns)}")

    # Group and count
    tidy = (
        df.groupby(resolved, dropna=dropna)
          .size()
          .reset_index(name=count_name)
    )

    # Sorting
    if sort:
        tidy = tidy.sort_values(resolved + [count_name]).reset_index(drop=True)

    # If canonical names differ from actual resolved names, optionally rename in output
    rename_back = {resolution_map[k]: k for k in groupby if resolution_map.get(k) not in (None, k)}
    if rename_back:
        tidy = tidy.rename(columns=rename_back)

    # Optionally build a wide pivot
    wide_df = None
    if wide:
        if pivot_index is None or pivot_columns is None:
            if len(groupby) >= 3:
                pivot_index = list(groupby[:2])
                pivot_columns = list(groupby[2:])
            else:
                raise ValueError("To create a wide table, please specify pivot_index and pivot_columns "
                                 "or provide at least 3 grouping columns.")

        def _resolve_list(names: Sequence[str]) -> List[str]:
            out = []
            for n in names:
                if n in tidy.columns:
                    out.append(n)
                else:
                    if n in resolution_map and resolution_map[n] in tidy.columns:
                        out.append(n if n in tidy.columns else resolution_map[n])
                    else:
                        candidates = alias_map.get(n, [])
                        chosen = next((c for c in candidates if c in tidy.columns), None)
                        if chosen is None:
                            raise ValueError(f"Pivot column '{n}' not found after resolution. "
                                             f"Available: {list(tidy.columns)}")
                        out.append(chosen)
            return out

        idx_cols = _resolve_list(list(pivot_index))
        col_cols = _resolve_list(list(pivot_columns))

        wide_df = (
            tidy.pivot_table(
                index=idx_cols,
                columns=col_cols,
                values=count_name,
                fill_value=0,
                aggfunc="sum",
            )
            .sort_index()
        )

    return tidy, wide_df
