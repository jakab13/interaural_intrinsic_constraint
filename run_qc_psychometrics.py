# Run QC (drops controls) and save pass/excluded CSVs
from Analysis.qc_psychometrics import run_qc_pipeline
import numpy as np
import pandas as pd

from Analysis.qc_psychometrics import run_qc_pipeline, export_all_subject_panels, filter_single_cue_subject_freq

# Build QC flags; drop controls; do NOT exclude outliers (so you can see everything)
df_pass, df_excl = run_qc_pipeline(
    "Dataframes/psychometrics_params.csv",
    drop_controls=True,
    write_outputs=False,
    apply_outlier_filter=False,
)

# Combine pass+excluded (since we didn’t filter outliers)
df_all = pd.concat([df_pass, df_excl], ignore_index=True)

singlecue_selection = {
    "kirke": 1400,
    "vp_1": 1700,
    "vp_2": 1600,
    "vp_3": 1700,
    "vp_4": 1300,
    "vp_9": 1400,
    "vp_10": 1500,
    "vp_11": 1700,
    "vp_12": 1600,
    "vp_13": 1800,
    "vp_14": 1200,
}


# Apply the single_cue subject→frequency selection
df_sel = filter_single_cue_subject_freq(df_all, singlecue_selection)

# (optional) save this filtered table for reference
df_sel.to_csv("Dataframes/psychometrics_params_selected.csv", index=False)
