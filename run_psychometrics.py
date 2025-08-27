# run_all_psychometrics.py  (or paste in a Python cell)

from psychometrics import run_and_persist_all_groups, DEFAULT_GROUPBY

PARAMS_OUT = "Dataframes/psychometrics_params.csv"

params = run_and_persist_all_groups(
    clean_csv="Dataframes/df_clean_master.csv",
    groupby=DEFAULT_GROUPBY,      # (dataset, subject_key, trial_type, standard_center_frequency, standard_angle_abs)
    overwrite=False,              # True = refit & overwrite existing JSON/PNGs
    include_freq_in_path=True,    # saves under Psychometrics/.../{f{freq}}/...
    save_fig=True,                # save PNGs alongside JSONs
    fig_dpi=300,
    shuffle=False,                # set True to randomize processing order
    limit=None,                   # set an int (e.g., 25) to test on first N groups
    params_out=PARAMS_OUT,
    progress_every=50,            # progress printout cadence
)

print(f"\n✅ Done. Saved params table to: {PARAMS_OUT}")
print(f"Rows: {len(params)} | Columns: {len(params.columns)}")
print("\nPreview:")
print(params.head(10))
