from pathlib import Path

# your modules
from postprocessing import (
    get_or_build_raw_master,
    apply_core_filters,
    per_file_row_counts,
    summarize_counts,  # re-exported from preprocessing
)

# (optional) schema if you want validation during building
# from preprocessing import SCHEMA_PSYCH_EXAMPLE

# One-time (recommended): force a rebuild so the 48-row/file QC is baked in
# df_raw = get_or_build_raw_master(
#     results_root="Results",
#     df_path="Dataframes/df_raw_master.csv",
#     rebuild=True,                                    # set True once to regenerate cleanly
#     loader_kwargs=dict(
#         sep="auto",
#         clean=True,
#         min_rows_per_file=12,
#         short_file="skip",
#         drop_columns=("inter_stimulus_interval", "mixing_gain"),
#         # schema=SCHEMA_PSYCH_EXAMPLE,   # uncomment to validate + coerce types
#         # schema_errors="warn",
#     ),
# )

# Later runs (fast): just load the cached CSV
df_raw = get_or_build_raw_master("Results", "Dataframes/df_raw_master.csv")

df_clean, reports = apply_core_filters(
    df_raw,
    allowed_freqs=(1300, 500),
    min_n_per_group=192,
    groupby=("dataset", "subject", "trial_type", "standard_center_frequency", "standard_angle_abs"),
    min_rows_per_file=12,          # will drop short files if any slipped into df_raw
    add_control_flag=True,
    angle_tolerance=1e-3,
)

# group counts before the 192-filter (can be <192)
reports["counts_pre"].head()

# groups removed by the 192-filter
reports["offenders"].head()

# group counts after the 192-filter (should all be >=192)
reports["counts_post"]["n"].min()  # expect >= 192

tidy, wide = summarize_counts(
    df_clean,
    groupby=["dataset", "subject", "trial_type", "standard_center_frequency", "standard_angle_abs", "is_control"]
)

tidy.head()

tidy[tidy.is_control == False].groupby(["dataset", "subject", "standard_center_frequency"]).count()["n_trials"]

exclusion_q = (df_clean.dataset == "combined_cue") & (df_clean.subject == "vp_6") & (df_clean.standard_angle_abs == 3.04)

df_clean_manual = df_clean[~exclusion_q]

df_clean_manual.to_csv("Dataframes/df_clean_master.csv")