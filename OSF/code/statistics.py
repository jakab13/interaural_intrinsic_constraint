from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

DF_DEFAULT = Path("OSF/data/psychometric_parameters.csv")

df = pd.read_csv(DF_DEFAULT)

df = df[df.reference_cue != df.comparison_cue]

df_single_cue = df[df.experiment == "single_cue"]
df_combined_cue_500 = df[(df.experiment == "combined_cue_500")]
df_combined_cue_1300 = df[(df.experiment == "combined_cue_1300")]



datasets = {
    "Single-cue": df_single_cue,
    "Combined-cue 500 Hz": df_combined_cue_500,
    "Combined-cue 1300 Hz": df_combined_cue_1300,
}

# These are the columns with squared prediction errors
PSE_UNCERT = "pse_pred_error_uncertainty"  # "pse_pred_error_uncertainty_squared"
PSE_SCAL   = "pse_pred_error_scaling"  # "pse_pred_error_scaling_squared"
JND_UNCERT = "jnd_pred_error_uncertainty"  # "jnd_pred_error_uncertainty_squared"
JND_SCAL   = "jnd_pred_error_scaling"  # "jnd_pred_error_scaling_squared"
SUBJ_COL   = "participant_id"


# --------------------------------------------------------
# 2. Function to run Wilcoxon tests per dataset
# --------------------------------------------------------
def run_wilcoxon_for_dataset(df):
    """
    df: dataframe for one dataset
    returns: dict with PSE and JND results,
             and the per-subject aggregated squared errors
    """

    # aggregate squared errors per subject (sum across conditions)
    agg = df.groupby(["participant_id", "trial_type"])[[PSE_UNCERT, PSE_SCAL, JND_UNCERT, JND_SCAL]].apply(lambda x: x.abs().mean())

    results = {}

    # PSE
    stat_pse, p_pse = wilcoxon(
        agg[PSE_UNCERT],
        agg[PSE_SCAL],
        alternative="greater"  # test: uncertainty error > scaling error
    )
    results["PSE"] = {
        "stat": stat_pse,
        "p": p_pse,
        "n_subjects": agg.shape[0],
        "median_uncert": agg[PSE_UNCERT].median(),
        "median_scal": agg[PSE_SCAL].median()
    }

    # JND
    stat_jnd, p_jnd = wilcoxon(
        agg[JND_UNCERT],
        agg[JND_SCAL],
        alternative="greater"
    )
    results["JND"] = {
        "stat": stat_jnd,
        "p": p_jnd,
        "n_subjects": agg.shape[0],
        "median_uncert": agg[JND_UNCERT].median(),
        "median_scal": agg[JND_SCAL].median()
    }

    return agg, results


# --------------------------------------------------------
# 3. Run tests for all datasets
# --------------------------------------------------------
all_results = {}

for label, df in datasets.items():
    agg, res = run_wilcoxon_for_dataset(df)
    all_results[label] = res


# --------------------------------------------------------
# 4. Nicely formatted output for the report
# --------------------------------------------------------
for dataset, res in all_results.items():
    print(f"\n=== {dataset} dataset ===")
    for measure, stats in res.items():
        W = stats["stat"]
        p = stats["p"]
        n = stats["n_subjects"]
        med_u = stats["median_uncert"]
        med_s = stats["median_scal"]

        print(
            f"{measure}: n = {n}, "
            f"median absolute error (uncertainty) = {med_u:.2f}, "
            f"median absolute error (scaling) = {med_s:.2f}, "
            f"W = {W}, p = {p}"
        )
