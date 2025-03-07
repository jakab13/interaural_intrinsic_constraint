# read in all the results files per subject
# generate one large "master" file that contains all the recordings of the experiments
# calculate the psychometric function for each participant, for each "trial_type", for each "standard_angle"
#       use the psignifit library for the calculations
# store the results (mean, slope) of the psychometric fit in the same data sheet
# plot psychometric functions in a Facetgrid using the seaborn library
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import numpy as np
import math
import psignifit as ps
import psignifit.psigniplot as psp
from scipy.stats import linregress
import statsmodels.api as sm
import matplotlib
matplotlib.use('MacOSX')

sns.set_style("white")

DIR = Path(os.getcwd())

# subject = "jakab_matched_single"
subject = "jakab"

folder_path = "/Users/jakabpilaszanovich/Documents/GitHub/interaural_intrinsic_constraint/Results/" + subject
# Get all CSV files and sort by modification time
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

combined_df = combined_df[(combined_df.stim_type == "noise_filtered_third_octave")]
# combined_df = combined_df[~(combined_df.comparison_angle == -2)]
# combined_df = combined_df[~(combined_df.comparison_angle == 2)]
# combined_df = combined_df[~(combined_df.standard_angle == 24)]
# combined_df = combined_df[~(combined_df.standard_center_frequency == 80)]
# combined_df = combined_df[~(combined_df.standard_center_frequency == 125)]
# combined_df = combined_df[~(combined_df.standard_center_frequency == 250)]
combined_df = combined_df[
    # (combined_df.standard_center_frequency == 400) |
    (combined_df.standard_center_frequency == 500) |
    # (combined_df.standard_center_frequency == 600) |
    # (combined_df.standard_center_frequency == 700) |
    (combined_df.standard_center_frequency == 800) |
    (combined_df.standard_center_frequency == 1000) |
    (combined_df.standard_center_frequency == 1200)
    # (combined_df.standard_center_frequency == 1500)
]
# combined_df = combined_df[
#     ~(combined_df.standard_angle == 1)
#     & ~(combined_df.standard_angle == 2)
#     & ~(combined_df.standard_angle == 3)
#     & ~(combined_df.standard_angle == 6)
#     & ~(combined_df.standard_angle == 12)
#     & ~(combined_df.standard_angle == 20)
# ]
# combined_df = combined_df[~(combined_df.standard_center_frequency == 1000)]
# combined_df = combined_df[~(combined_df.trial_type == "BOTH-->BOTH")]

def get_cue_combination(row):
    trial_type_string = row.trial_type.split("-->")
    from_type = trial_type_string[0]
    to_type = trial_type_string[-1]
    if from_type == to_type:
        cue_combination = "within"
    else:
        cue_combination = "across"
    if from_type == "BOTH" or to_type == "BOTH" and from_type != to_type:
        cue_combination = "combined"
    return cue_combination

combined_df["cue_combination"] = combined_df.apply(lambda row: get_cue_combination(row), axis=1)

# g = sns.FacetGrid(combined_df, hue="trial_type", row="standard_angle", col="standard_center_frequency", height=2,
#                   hue_order=["ILD-->ILD", "ITD-->ITD", "ITD-->ILD", "ILD-->ITD", "BOTH-->BOTH", "BOTH-->ITD", "BOTH-->ILD"])
# g.refline(y=0.5, c="grey", alpha=.1)
# g.refline(x=0, c="grey", alpha=.1)
# [ax.axvline(float(ax.title.get_text().split("standard_angle = ")[1].split("|")[0]), ymax=0.5, c="black") for ax in g.axes.flatten()]
# g.map(sns.lineplot, "comparison_angle", "score", marker="o")
# g.set_titles(template="Standard: {row_name}° at {col_name}Hz ")
# g.add_legend()


# g = sns.FacetGrid(combined_df, hue="trial_type", row="standard_angle", col="standard_center_frequency", height=2, aspect=2,
#                   hue_order=["ILD-->ILD", "ITD-->ITD",
#                              "ITD-->ILD", "ILD-->ITD"
#                              ], row_order=[0, 4, 8, 16], col_order=[800, 1000, 1200])
# g.refline(y=0.5, c="grey", alpha=.1)
# g.refline(x=0, c="grey", alpha=.1)
# [ax.axvline(float(ax.title.get_text().split("standard_angle = ")[1].split("|")[0]), ymax=0.5, c="black") for ax in g.axes.flatten()]
# g.map(sns.regplot, "comparison_angle", "score",
#       scatter=False,
#       logistic=True,
#       ci=None
#       )
# g.set_titles(template="Standard: {row_name}° at {col_name}Hz ")
# g.axes[3,0].set_xlabel('Comparison angle')
# g.axes[3,1].set_xlabel('Comparison angle')
# g.axes[3,2].set_xlabel('Comparison angle')
# g.set_ylabels("Perceived 'right'")
# g.set(xlim=(-40, 40))
# g.add_legend()

# g = sns.FacetGrid(combined_df,
#                   hue="trial_type", height=4, col="standard_center_frequency",
#                 #   hue_order=["BOTH-->BOTH", "ITD-->BOTH", "ITD-->ITD", "BOTH-->ITD"],
#                 # palette=sns.color_palette("Paired")[6:],
#                 hue_order=["ILD-->ILD", "ITD-->ILD", "ITD-->ITD", "ILD-->ITD"],
#                   palette=sns.color_palette("Paired"),
#                   row="stim_type"
#                   )
# g.refline(y=0.5, c="grey", alpha=.1)
# g.refline(x=0, c="grey", alpha=.1)
# g.map(sns.lineplot, "comparison_angle", "score", marker="o",
#       errorbar=None
#       )
# # g.map(sns.regplot, "comparison_angle", "score", logistic=True, ci=None, scatter=False)
# g.add_legend()
# # plt.savefig("Cue matching - combined cue (sharp onset)", dpi=200)
# # plt.savefig("Cue matching - single cue", dpi=200)


# g = sns.FacetGrid(combined_df,
#                   hue="trial_type", col="standard_center_frequency",
#                 hue_order=["ILD-->ILD", "ITD-->ITD"], row="standard_angle"
#                   )
# g.refline(y=0.5, c="grey", alpha=.1)
# g.refline(x=0, c="grey", alpha=.1)
# g.map(sns.lineplot, "comparison_angle", "score", marker="o",
#       errorbar=None
#       )
# g.add_legend()


def add_line_of_equality(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def add_grid_line_of_equality(g):
    for ax in g.axes.flatten():
        add_line_of_equality(ax, color="black", alpha=.1)

def get_psychometric_means(g, save_fig=False):
    subject = g["subject"].unique()[0]
    standard_angle = g["standard_angle"].unique()[0]
    standard_center_frequency = g["standard_center_frequency"].unique()[0]
    stim_type = g["stim_type"].unique()[0]
    trial_type = g["trial_type"].unique()[0]
    title = f"{subject}_{standard_center_frequency}_{trial_type}_{int(standard_angle)}"
    ps_result_file_path = DIR / Path("ps_results") / Path(subject) / Path(title + ".json")
    data = g.groupby("comparison_angle", as_index=False).agg(
        {"score": "sum", g.columns[0]: "count"}).rename(
        columns={g.columns[0]: 'n_total'})
    if ps_result_file_path.exists():
        res = ps.Result.load_json(ps_result_file_path)
        print(f"Loaded psignifit estimates for {title}")
        if not np.array_equal(res.data, data.values):
            res = ps.psignifit(
                data.values,
                sigmoid="logistic",
                experiment_type="equal asymptote"
            )
            print(f"Calculated psignifit estimates for {title}")
            res.save_json(ps_result_file_path)
            if save_fig:
                path = "figures/" + subject + "/"  + title
                plt.figure()
                psp.plot_psychometric_function(res)
                plt.savefig(path)
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
            path = "figures/" + subject + "/" + title
            plt.figure()
            psp.plot_psychometric_function(res)
            plt.savefig(path)
            plt.close()

    PSE = res.parameters_estimate_mean["threshold"]
    PSE_ci_95_low = res.confidence_intervals['threshold']["0.95"][0]
    PSE_ci_95_high = res.confidence_intervals['threshold']["0.95"][1]
    # print(res)

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

df_group = combined_df.groupby(["subject", "standard_angle", "standard_center_frequency", "trial_type", "stim_type"])
df_model = df_group.apply(lambda g: get_psychometric_means(g, save_fig=True), include_groups=True).reset_index()

# df_model = pd.read_csv("PSE_estimates_kirke.csv")
# df_model = df_model[~(df_model.standard_center_frequency == 1000)]
# df_model = df_model[~(df_model.standard_angle == 0)]

def find_base_JND(row, cue_type="ITD"):
    standard_center_frequency = row["standard_center_frequency"]
    standard_angle = row["standard_angle"]
    base = df_model[(df_model.standard_center_frequency == standard_center_frequency) &
                    (df_model.standard_angle == standard_angle) &
                    (df_model.trial_type == str(cue_type + "-->" + cue_type))]
    if len(base) > 0:
        base_JND = base.JND.values[0]
    else:
        base_JND = None
    return base_JND

def find_base_PSE(row, cue_type="ITD"):
    standard_center_frequency = row["standard_center_frequency"]
    standard_angle = row["standard_angle"]
    base = df_model[(df_model.standard_center_frequency == standard_center_frequency) &
                    (df_model.standard_angle == standard_angle) &
                    (df_model.trial_type == str(cue_type + "-->" + cue_type))]
    if len(base) > 0:
        base_PSE = base.PSE.values[0]
    else:
        base_PSE = None
    return base_PSE

df_model["base_ITD_JND"] = df_model.apply(lambda row: find_base_JND(row, cue_type="ITD"), axis=1)
df_model["base_ILD_JND"] = df_model.apply(lambda row: find_base_JND(row, cue_type="ILD"), axis=1)

df_model["base_ITD_PSE"] = df_model.apply(lambda row: find_base_PSE(row, cue_type="ITD"), axis=1)
df_model["base_ILD_PSE"] = df_model.apply(lambda row: find_base_PSE(row, cue_type="ILD"), axis=1)

# for from_type in ["ILD", "BOTH"]:
#     df_repeat = df_model[(df_model["trial_type"] == "ITD-->ITD") & (df_model["standard_angle"] == 0)]
#     curr_trial_type = f"{from_type}-->ITD"
#     df_repeat.loc[:, "trial_type"] = curr_trial_type
#     df_model = pd.concat([df_repeat, df_model])
#
# for from_type in ["ITD", "BOTH"]:
#     df_repeat = df_model[(df_model["trial_type"] == "ILD-->ILD") & (df_model["standard_angle"] == 0)]
#     curr_trial_type = f"{from_type}-->ILD"
#     df_repeat.loc[:, "trial_type"] = curr_trial_type
#     df_model = pd.concat([df_repeat, df_model])


def get_regressions(g):
    # reg_PSE = sm.OLS(exog=g["standard_angle"].values, endog=g["PSE"].values).fit()
    reg_PSE = linregress(g["standard_angle"].values, g["PSE"].values)
    reg_JND = linregress(g["standard_angle"].values, g["JND"].values)
    reg_corr = linregress(g["PSE"].values, g["JND"].values)
    row = {
        # "PSE_slope": reg_PSE.params[0],
        # "PSE_intercept": 0,
        # "PSE_pvalue": reg_PSE.pvalues[0],
        "PSE_slope": reg_PSE.slope,
        "PSE_intercept": reg_PSE.intercept,
        "PSE_pvalue": reg_PSE.pvalue,
        "JND_slope": reg_JND.slope,
        "JND_intercept": reg_JND.intercept,
        "JND_pvalue": reg_JND.pvalue,
        "PSE_vs_JNDS_slope": reg_corr.slope,
        "PSE_vs_JNDS_intercept": reg_corr.intercept,
        "PSE_vs_JNDS_pvalue": reg_corr.pvalue,
    }
    return pd.Series(row)

df_model_regs = df_model.groupby(["subject", "standard_center_frequency", "trial_type"])\
    .apply(get_regressions).reset_index()

df_model = pd.merge(left=df_model, right=df_model_regs, on=["subject", "standard_center_frequency", "trial_type"])


def get_PSE_pred_reg(row):
    return row["standard_angle"] * row["PSE_slope"] + row["PSE_intercept"]


def get_JND_pred_reg(row):
    return row["standard_angle"] * row["JND_slope"] + row["JND_intercept"]

def get_PSE_pred_IC(row):
    trial_type_string = row.trial_type.split("-->")
    from_type = trial_type_string[0]
    to_type = trial_type_string[-1]
    if from_type == "BOTH":
        PSE_pred = None
    elif to_type == "BOTH":
        PSE_pred = None
    else:
        from_base_name = "base_" + from_type + "_JND"
        to_base_name = "base_" + to_type + "_JND"
        from_base_name_PSE = "base_" + from_type + "_PSE"
        to_base_name_PSE = "base_" + to_type + "_PSE"
        from_base_JND = row[from_base_name]
        to_base_JND = row[to_base_name]
        from_base_PSE = row[from_base_name_PSE]
        to_base_PSE = row[to_base_name_PSE]
        PSE_pred = row.standard_angle * (to_base_JND / from_base_JND)
    return PSE_pred

def get_PSE_pred_Bayes(row):
    trial_type_string = row.trial_type.split("-->")
    from_type = trial_type_string[0]
    to_type = trial_type_string[-1]
    if from_type == "BOTH" and to_type != "BOTH":
        to_index = ["ILD", "ITD"].index(to_type)
        from_index = 1 - to_index
        from_type_2 = ["ILD", "ITD"][from_index]
        from_base_name = "base_" + from_type_2 + "_JND"
        to_base_name = "base_" + to_type + "_JND"
        from_base_JND = row[from_base_name]
        to_base_JND = row[to_base_name]
        from_base_sigma = from_base_JND / math.sqrt(2)
        to_base_sigma = to_base_JND / math.sqrt(2)
        sigma_combined_squared = ((to_base_sigma ** 2) * (from_base_sigma ** 2)) / (
                    (to_base_sigma ** 2) + (from_base_sigma ** 2))
        PSE_pred = row.standard_angle * (to_base_sigma ** 2) / sigma_combined_squared
    elif to_type == "BOTH":
        PSE_pred = None
    else:
        from_base_name = "base_" + from_type + "_JND"
        to_base_name = "base_" + to_type + "_JND"
        from_base_JND = row[from_base_name]
        to_base_JND = row[to_base_name]
        from_base_sigma = from_base_JND / math.sqrt(2)
        to_base_sigma = to_base_JND / math.sqrt(2)
        PSE_pred = row.standard_angle * ((to_base_sigma ** 2) / (from_base_sigma ** 2))
    return PSE_pred


def get_JND_pred_Bayes(row):
    trial_type_string = row.trial_type.split("-->")
    from_type = trial_type_string[0]
    to_type = trial_type_string[-1]
    if from_type == "BOTH" and to_type != "BOTH":
        to_index = ["ILD", "ITD"].index(to_type)
        from_index = 1 - to_index
        from_type = ["ILD", "ITD"][from_index]
        from_base_name = "base_" + from_type + "_JND"
        to_base_name = "base_" + to_type + "_JND"
        from_base_JND = row[from_base_name]
        to_base_JND = row[to_base_name]
        from_base_sigma = from_base_JND / math.sqrt(2)
        to_base_sigma = to_base_JND / math.sqrt(2)
        angle = row.standard_angle
        sigma_combined_squared = ((to_base_sigma ** 2) * (from_base_sigma ** 2)) / ((to_base_sigma ** 2) + (from_base_sigma ** 2))
        sigma_combined = math.sqrt(sigma_combined_squared)
        PSE_term = angle * sigma_combined_squared / (to_base_sigma ** 2)
        constant_term = sigma_combined
        JND_pred = PSE_term + constant_term
    elif to_type == "BOTH":
        JND_pred = None
    else:
        from_base_name = "base_" + from_type + "_JND"
        to_base_name = "base_" + to_type + "_JND"
        from_base_JND = row[from_base_name]
        to_base_JND = row[to_base_name]
        from_base_sigma = from_base_JND / math.sqrt(2)
        to_base_sigma = to_base_JND / math.sqrt(2)
        angle = row.standard_angle
        sigma_combined_squared = ((to_base_sigma ** 2) * (from_base_sigma ** 2) / ((to_base_sigma ** 2) + (from_base_sigma ** 2)))
        sigma_combined = math.sqrt(sigma_combined_squared)
        PSE_term = angle * ((to_base_sigma ** 2) / (from_base_sigma ** 2))
        constant_term = (to_base_sigma ** 2) / sigma_combined
        JND_pred = PSE_term + constant_term
    return JND_pred

def get_JND_pred_IC(row):
    trial_type_string = row.trial_type.split("-->")
    from_type = trial_type_string[0]
    to_type = trial_type_string[-1]
    if from_type == "BOTH":
        JND_pred = None
    elif to_type == "BOTH":
        JND_pred = None
    else:
        from_base_name = "base_" + from_type + "_JND"
        to_base_name = "base_" + to_type + "_JND"
        from_base_JND = row[from_base_name]
        to_base_JND = row[to_base_name]
        from_base_sigma = from_base_JND / math.sqrt(2)
        to_base_sigma = to_base_JND / math.sqrt(2)
        constant_term = (to_base_sigma ** 2) * math.sqrt(
            ((to_base_sigma ** 2) + (from_base_sigma ** 2)) / ((to_base_sigma ** 2) * (from_base_sigma ** 2)))
        JND_pred = constant_term + row.PSE - row.standard_angle
    return JND_pred

# df_model["PSE_pred_reg"] = df_model.apply(lambda row: get_PSE_pred_reg(row), axis=1)
# df_model["PSE_pred_IC"] = df_model.apply(lambda row: get_PSE_pred_IC(row), axis=1)
df_model["PSE_pred_Bayes"] = df_model.apply(lambda row: get_PSE_pred_Bayes(row), axis=1)
# df_model["JND_pred_reg"] = df_model.apply(lambda row: get_JND_pred_reg(row), axis=1)
df_model["JND_pred_IC"] = df_model.apply(lambda row: get_JND_pred_IC(row), axis=1)
df_model["JND_pred_Bayes"] = df_model.apply(lambda row: get_JND_pred_Bayes(row), axis=1)

df_model["PSE_delta"] = df_model["PSE"] - df_model["standard_angle"]

df_model["cue_combination"] = df_model.apply(lambda row: get_cue_combination(row), axis=1)

def show_reg_vars(slope, intercept, **kwargs):
    l = kwargs["label"]
    to_type = l.split("-->")[-1]
    y_pos = 38
    if to_type == "ITD":
        y_pos = 33
    elif to_type == "BOTH":
        y_pos = 28
    intercept = round(intercept.mean(), 1)
    intercept_text = f"+{intercept}" if intercept > 0 else f"{intercept}"
    text = f'y={round(slope.mean(), 1)}x{intercept_text}'
    plt.text(0, y_pos, text, fontsize=11, c=kwargs["color"])


def show_CIs(standard_angle, ci_high, ci_low, **kwargs):
    plt.fill_between(standard_angle, ci_high, ci_low, alpha=.1, **kwargs)

# PSE plot
# df_curr = df_model[df_model.standard_angle > 0]
df_curr = df_model
# df_curr = df_curr[~(df_curr.standard_angle == 1)]
# df_curr = df_curr[~(df_curr.standard_angle == 3)]
# df_curr = df_curr[~(df_curr.standard_angle == 6)]
df_curr = df_curr[~(df_curr.standard_angle == 20)]

g = sns.FacetGrid(
    data=df_curr,
    col="standard_center_frequency",
    col_order=[500, 800, 1000, 1200],
    row="cue_combination",
    row_order=["within", "across", "combined"],
    hue="trial_type",
    hue_order=[
        "ILD-->ILD", "ITD-->ITD",
        "ITD-->ILD", "ILD-->ITD",
        "BOTH-->ITD", "BOTH-->ILD",
        "BOTH-->BOTH"
    ],
    palette=sns.color_palette("tab10"),
    height=2.5
)
# g.map(show_reg_vars, "PSE_slope", "PSE_intercept")
# g.map(show_CIs, "standard_angle", "PSE_ci_95_high", "PSE_ci_95_low")
g.map(sns.lineplot, "standard_angle", "PSE", marker="o")
# g.map(sns.regplot, "standard_angle", "PSE_pred_Bayes",  scatter=False, ci=None, line_kws={"ls": "--"})
g.set_titles(template="{col_name}Hz ({row_name} cue)")
add_grid_line_of_equality(g)
g.axes[0,0].set_xlabel('Physical angle')
g.axes[0,1].set_xlabel('Physical angle')
g.axes[0,2].set_xlabel('Physical angle')
g.set_ylabels("PSE")
g.add_legend()
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Points of Subjective Equivalence')
# plt.savefig("PSEs - all cue combinations.png", dpi=200)


# JND plot
g = sns.FacetGrid(
    data=df_curr,
    col="standard_center_frequency",
    col_order=[500, 800, 1200],
    row="cue_combination",
    row_order=["within", "across", "combined"],
    hue="trial_type",
    hue_order=[
        "ILD-->ILD", "ITD-->ITD",
        "ITD-->ILD", "ILD-->ITD",
        "BOTH-->ITD", "BOTH-->ILD",
        "BOTH-->BOTH"
    ],
    palette=sns.color_palette("tab10"),
    height=2.5
)
# g.map(show_reg_vars, "JND_slope", "JND_intercept")
# g.map(show_CIs, "standard_angle", "JND_ci_95_high", "JND_ci_95_low")
# g.map(sns.regplot, "standard_angle", "JND", ci=None)
g.map(sns.scatterplot, "standard_angle", "JND")
# g.map(sns.regplot, "standard_angle", "JND_pred_Bayes", scatter=False, ci=None, line_kws={"ls": "--"})
# g.map(sns.regplot, "standard_angle", "JND_pred_IC", scatter=False, ci=None, line_kws={"ls": "-"})
g.set_titles(template="{col_name}Hz ")
g.axes[0,0].set_xlabel('Physical angle')
g.axes[0,1].set_xlabel('Physical angle')
g.axes[0,2].set_xlabel('Physical angle')
g.set_ylabels("JND")
g.set(ylim=(0, 50))
g.add_legend()
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('JNDs')
# plt.savefig("JNDs - all cue combinations.png", dpi=200)

# PSE vs JND plot
# g = sns.FacetGrid(
#     data=df_curr,
#     row="cue_combination",
#     hue="trial_type",
#     col="standard_center_frequency",
#     col_order=[500, 800, 1000, 1200],
#     palette=sns.color_palette("tab10")[2:],
#     height=2.5,
#     hue_order=["ITD-->ILD", "ILD-->ITD", "ITD-->BOTH", "ILD-->BOTH"],
#     row_order=["across", "combined"]
# )
# g.map(show_reg_vars, "PSE_vs_JNDS_slope", "PSE_vs_JNDS_intercept")
# # g.map(show_CIs, "PSE", "JND_ci_95_high", "JND_ci_95_low")
# g.map(sns.regplot, "PSE", "JND", ci=None)
# g.set_titles(template="{col_name}Hz ")
# # g.map(sns.lineplot, "standard_angle", "JND_pred_IC", errorbar=None, label="predicted", marker="o")
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle('PSEs vs JNDs')
# g.add_legend()
# plt.savefig("PSEs vs JNDs - across and combined cues.png", dpi=200)

# MATCHING ==============================================

# freq = 800
# ILD_angle = 8
# ITD_angle = df_model[(df_model.standard_center_frequency == freq)
#                      & (df_model.trial_type == "ILD-->ITD")
#                      & (df_model.standard_angle == ILD_angle)]["PSE"].values[0]
#
# reg_ILD_ILD = df_model_regs[(df_model_regs.standard_center_frequency == freq)
#                      & (df_model_regs.trial_type == "ILD-->ILD")]
# JND_ILD_ILD = reg_ILD_ILD["JND_slope"].values[0] * ILD_angle + reg_ILD_ILD["JND_intercept"].values[0]
#
# reg_ILD_ITD = df_model_regs[(df_model_regs.standard_center_frequency == freq)
#                      & (df_model_regs.trial_type == "ILD-->ITD")]
# JND_ILD_ITD = reg_ILD_ITD["JND_slope"].values[0] * ILD_angle + reg_ILD_ITD["JND_intercept"].values[0]
#
# reg_ITD_ITD = df_model_regs[(df_model_regs.standard_center_frequency == freq)
#                      & (df_model_regs.trial_type == "ITD-->ITD")]
# JND_ITD_ITD = reg_ITD_ITD["JND_slope"].values[0] * ITD_angle + reg_ITD_ITD["JND_intercept"].values[0]
#
# reg_ITD_ILD = df_model_regs[(df_model_regs.standard_center_frequency == freq)
#                      & (df_model_regs.trial_type == "ITD-->ILD")]
# JND_ITD_ILD = reg_ITD_ILD["JND_slope"].values[0] * ITD_angle + reg_ITD_ILD["JND_intercept"].values[0]
#
# sns.catplot([JND_ITD_ITD, JND_ILD_ITD, JND_ILD_ILD, JND_ITD_ILD])

# sns.catplot(data=df_model, kind="bar", x="JND", hue="trial_type", hue_order=["ILD-->ILD", "ITD-->ILD", "ITD-->ITD", "ILD-->ITD"], palette=sns.color_palette("Paired"), col="standard_center_frequency")
