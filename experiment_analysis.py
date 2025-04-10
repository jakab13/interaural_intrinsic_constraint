import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import numpy as np
import psignifit as ps
import psignifit.psigniplot as psp
import matplotlib
matplotlib.use('MacOSX')

sns.set_style("white")

DIR = Path(os.getcwd())


def get_df(subject):
    folder_path = DIR / "Results" / subject
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    csv_files = sorted(
        csv_files,
        key=lambda x: os.path.getmtime(os.path.join(folder_path, x))
    )
    # Load and combine the data into a single DataFrame
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return combined_df


def get_psychometric_estimates(g, save_fig=False):
    subject = g["subject"].unique()[0]
    standard_angle = g["standard_angle_abs"].unique()[0]
    standard_center_frequency = g["standard_center_frequency"].unique()[0]
    comparison_center_frequency = g["comparison_center_frequency"].unique()[0]
    trial_type = g["trial_type"].unique()[0]
    title = f"{subject}_{standard_center_frequency}-->{comparison_center_frequency}_{trial_type}_{int(standard_angle)}"
    ps_result_folder_path = DIR / Path("ps_results") / Path(subject)
    Path(ps_result_folder_path).mkdir(parents=True, exist_ok=True)
    ps_result_file_path = ps_result_folder_path / Path(title + ".json")
    data = g.groupby("comparison_angle_abs", as_index=False).agg(
        {"score_abs": "sum", g.columns[0]: "count"}).rename(
        columns={g.columns[0]: 'n_total'})
    if ps_result_file_path.exists():
        res = ps.Result.load_json(ps_result_file_path)
        # print(f"Loaded psignifit estimates for {title}")
        if not np.array_equal(res.data, data.values):
            res = ps.psignifit(
                data.values,
                sigmoid="logistic",
                experiment_type="equal asymptote"
            )
            print(f"Calculated psignifit estimates for {title}")
            res.save_json(ps_result_file_path)
            if save_fig:
                ps_figure_file_path = "figures/" + subject + "/" + title
                Path(ps_figure_file_path).mkdir(parents=True, exist_ok=True)
                plt.figure()
                psp.plot_psychometric_function(res)
                plt.savefig(ps_figure_file_path)
                plt.close()
    else:
        res = ps.psignifit(
            data.values,
            sigmoid="logistic",
            experiment_type="equal asymptote"
        )
        print(f"Calculated psignifit estimates for {title}")
        res.save_json(ps_result_file_path)
        if save_fig:
            ps_figure_file_path = "figures/" + subject + "/" + title
            Path(ps_figure_file_path).mkdir(parents=True, exist_ok=True)
            plt.figure()
            psp.plot_psychometric_function(res)
            plt.savefig(ps_figure_file_path)
            plt.close()

    PSE = res.parameters_estimate_mean["threshold"]
    PSE_ci_95_low = res.confidence_intervals['threshold']["0.95"][0]
    PSE_ci_95_high = res.confidence_intervals['threshold']["0.95"][1]

    JND = res.parameters_estimate_mean["width"] / 2
    JND_ci_95_low = res.confidence_intervals['width']["0.95"][0] / 2
    JND_ci_95_high = res.confidence_intervals['width']["0.95"][1] / 2
    slope = res.slope_at_proportion_correct(0.5)
    row = {
        "PSE": PSE,
        "PSE_ci_95_low": PSE_ci_95_low,
        "PSE_ci_95_high": PSE_ci_95_high,
        "JND": JND,
        "JND_ci_95_low": JND_ci_95_low,
        "JND_ci_95_high": JND_ci_95_high,
        "slope": slope
    }
    return pd.Series(row)


def get_model_table(subject):
    combined_df = get_df(subject)
    df_group = combined_df.groupby(["subject", "standard_angle_abs", "standard_center_frequency", "trial_type"])
    df_model = df_group.apply(lambda g: get_psychometric_estimates(g, save_fig=True), include_groups=True).reset_index()
    return df_model


def get_PSE(subject, standard_cue, comparison_cue, standard_angle_abs, standard_center_frequency):
    trial_type = standard_cue + "-->" + comparison_cue
    df_model = get_model_table(subject)
    PSE_row = df_model[(df_model.subject == subject)
                   & (df_model.standard_angle_abs == standard_angle_abs)
                   & (df_model.trial_type == trial_type)
                   & (df_model.standard_center_frequency == standard_center_frequency)]
    PSE = PSE_row["PSE"].values[0]
    return PSE


def plot_pfs(subject):
    get_model_table(subject)
    folder_path = DIR / "ps_results" / subject
    ps_result_file_paths = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    colors = {
        "ITD-->ITD": "tab:orange",
        "ILD-->ITD": "tab:red",
        "ILD-->ILD": "tab:blue",
        "ITD-->ILD": "tab:green"
    }
    fig, ax = plt.subplots(1, 1)
    for ps_result_file_path in ps_result_file_paths:
        res = ps.Result.load_json(folder_path / ps_result_file_path)
        standard_cue = ps_result_file_path.split("-->")[1][-3:]
        comparison_cue = ps_result_file_path.split("-->")[2][:3]
        trial_type = standard_cue + "-->" + comparison_cue
        psp.plot_psychometric_function(res,
                                       ax=ax,
                                       data_color=colors[trial_type],
                                       line_color=colors[trial_type],
                                       extrapolate_stimulus=0
        )