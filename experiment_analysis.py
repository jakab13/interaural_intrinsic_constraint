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

    JND = res.threshold(0.84, return_ci=False) - PSE
    slope = res.slope_at_proportion_correct(0.5)
    row = {
        "PSE": PSE,
        "PSE_ci_95_low": PSE_ci_95_low,
        "PSE_ci_95_high": PSE_ci_95_high,
        "JND": JND,
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


def plot_psychometric_function_local(result,  # noqa: C901, this function is too complex
                               ax: matplotlib.axes.Axes = None,
                               plot_data: bool = True,
                               plot_parameter: bool = True,
                               plot_JND: bool = False,
                               data_color: str = '#0069AA',  # blue
                               data_size: float = 1.,
                               line_color: str = '#000000',  # black
                               line_width: float = 2,
                               line_style: str = "-",
                               extrapolate_stimulus: float = 0.2,
                               x_label='Stimulus Level',
                               y_label='Proportion Correct',
                                **line_kws):
    """ Plot psychometric function fit together with the data.
    """
    if ax is None:
        ax = plt.gca()

    params = result.get_parameters_estimate(estimate_type="MAP")
    data = np.asarray(result.data)
    config = result.configuration

    if params['gamma'] is None:
        params['gamma'] = params['lambda']
    if len(data) == 0:
        return
    # data_size = min(20, 10000. / np.sum(data[:, 2]))

    ymin = 0

    x_data = data[:, 0]
    if plot_data:
        y_data = data[:, 1] / data[:, 2]
        # the size is proportional to the sqrt of the data size, as in the MATLAB version.
        # We added a factor of 10 to make visually similar to the MATLAB version
        size = np.sqrt(data_size / data[:, 2]) * 1000 * data_size
        # size = np.sqrt(data[:, 2]) * ( * 100)
        ax.scatter(x_data, y_data, s=size, color=data_color, marker='o', clip_on=False)

    sigmoid = config.make_sigmoid()
    x = np.linspace(x_data.min(), x_data.max(), num=1000)
    x_low = np.linspace(x[0] - extrapolate_stimulus * (x[-1] - x[0]), x[0], num=100)
    x_high = np.linspace(x[-1], x[-1] + extrapolate_stimulus * (x[-1] - x[0]), num=100)
    y = sigmoid(np.r_[x_low, x, x_high], params['threshold'], params['width'])
    y = (1 - params['gamma'] - params['lambda']) * y + params['gamma']
    ax.plot(x, y[len(x_low):-len(x_high)], c=line_color, lw=line_width, ls=line_style, **line_kws, clip_on=False)
    ax.plot(x_low, y[:len(x_low)], '--', c=line_color, lw=line_width, clip_on=False)
    ax.plot(x_high, y[-len(x_high):], '--', c=line_color, lw=line_width, clip_on=False)

    if plot_parameter:
        x = [params['threshold'], params['threshold']]
        y = [ymin, params['gamma'] + (1 - params['lambda'] - params['gamma']) * config.thresh_PC]
        ax.plot(x, y, '-', c=line_color)

        # ax.axhline(y=1 - params['lambda'], linestyle=':', color=line_color)
        # ax.axhline(y=params['gamma'], linestyle=':', color=line_color)

        CI_95 = result.confidence_intervals['threshold']['0.95']
        y = np.array([params['gamma'] + .5 * (1 - params['lambda'] - params['gamma'])] * 2)
        ax.plot(CI_95, y, c=line_color)
        ax.plot([CI_95[0]] * 2, y + [-.01, .01], c=line_color)
        ax.plot([CI_95[1]] * 2, y + [-.01, .01], c=line_color)

    if plot_JND:
        JND = result.threshold(.84, return_ci=False)
        ax.vlines(x=JND, ymin=0, ymax=0.84, color=line_color, ls=line_style)

    # AXIS SETTINGS
    plt.axis('tight')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_ylim([ymin, 1])
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_pfs(subject, plot_parameter=True, plot_JND=False):
    df = get_model_table(subject)
    folder_path = DIR / "ps_results" / subject
    ps_result_file_paths = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    cmap = matplotlib.cm.get_cmap('tab20')
    line_kws = {
        "ITD-->ITD": {"color": cmap(2), "ls": "-"},
        "ILD-->ITD": {"color": cmap(3), "ls": "--"},
        "ILD-->ILD": {"color": cmap(0), "ls": "-"},
        "ITD-->ILD": {"color": cmap(1), "ls": "--"}
    }
    fig, (ax_0, ax_1) = plt.subplots(1, 2, sharey=False, width_ratios=[3, 1], figsize=(16, 6))
    ax_0.axhline(y=0.5, color="lightgrey", alpha=.3)
    if plot_JND:
        ax_0.axhline(y=0.84, color="lightgrey", alpha=.3)
    for ps_result_file_path in ps_result_file_paths:
        res = ps.Result.load_json(folder_path / ps_result_file_path)
        standard_cue = ps_result_file_path.split("-->")[1][-3:]
        comparison_cue = ps_result_file_path.split("-->")[2][:3]
        trial_type = standard_cue + "-->" + comparison_cue
        plot_psychometric_function_local(res,
                                         ax=ax_0,
                                         plot_parameter=plot_parameter,
                                         plot_JND=plot_JND,
                                         data_color=line_kws[trial_type]["color"],
                                         data_size=.5,
                                         extrapolate_stimulus=0,
                                         line_style=line_kws[trial_type]["ls"],
                                         line_color=line_kws[trial_type]["color"],
                                         label=trial_type)

    ax_0.legend()

    sns.barplot(df, ax=ax_1, x="trial_type", y="JND", hue="trial_type",
                hue_order=["ILD-->ILD", "ITD-->ILD", "ITD-->ITD", "ILD-->ITD"],
                order=["ILD-->ILD", "ITD-->ILD", "ITD-->ITD", "ILD-->ITD"],
                palette="tab20")
    ax_1.set_ylim(0, df["JND"].max() * 1.05)
    freq = df.standard_center_frequency.unique()[0]
    title = f"{subject} at {freq}Hz"
    fig.suptitle(title, fontsize=14)


