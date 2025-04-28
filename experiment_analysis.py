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


def ask_to_continue(prompt="Do you want to continue? (y): ", expected="y"):
    while True:
        response = input(prompt).strip().lower()
        if response == expected:
            break
        print(f"Please type '{expected}' to continue.")


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
    ps_result_folder_path = DIR / Path("ps_results") / Path(subject) / f"{standard_center_frequency}-->{comparison_center_frequency}"
    Path(ps_result_folder_path).mkdir(parents=True, exist_ok=True)
    ps_result_file_path = ps_result_folder_path / Path(title + "°" + ".json")
    data = g.groupby("comparison_angle_abs", as_index=False).agg(
        {"score_abs": "sum", g.columns[0]: "count"}).rename(
        columns={g.columns[0]: 'n_total'})
    if ps_result_file_path.exists():
        res = ps.Result.load_json(ps_result_file_path)
        # print(f"Loaded psignifit estimates for {title}")
        if not np.array_equal(res.data, data.values):
            res = ps.psignifit(
                data.values,
                sigmoid="norm",
                experiment_type="equal asymptote"
                # cuts=[0.5, 0.84]
                # bootstrapMethod="parametric",
                # nBootstrapSamples=1000
            )
            print(f"Calculated psignifit estimates for {title}")
            res.save_json(ps_result_file_path)
            if save_fig:
                ps_figure_folder_path = DIR / Path("figures") / Path(subject) / f"{standard_center_frequency}-->{comparison_center_frequency}"
                ps_figure_file_path = ps_figure_folder_path / Path(title + "°")
                Path(ps_figure_folder_path).mkdir(parents=True, exist_ok=True)
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
            ps_figure_folder_path = DIR / Path("figures") / Path(
                subject) / f"{standard_center_frequency}-->{comparison_center_frequency}"
            ps_figure_file_path = ps_figure_folder_path / Path(title + "°")
            Path(ps_figure_folder_path).mkdir(parents=True, exist_ok=True)
            plt.figure()
            psp.plot_psychometric_function(res)
            plt.savefig(ps_figure_file_path)
            plt.close()

    PSE = res.parameters_estimate_mean["threshold"]
    PSE_ci_95_low = res.confidence_intervals['threshold']["0.95"][0]
    PSE_ci_95_high = res.confidence_intervals['threshold']["0.95"][1]

    JND = res.threshold(0.84, return_ci=False, unscaled=False) - PSE

    width = res.parameters_estimate_mean["width"] / 2
    width_ci_95_low = res.confidence_intervals['width']["0.95"][0] / 2
    width_ci_95_high = res.confidence_intervals['width']["0.95"][1] / 2

    JND_ci_95_low = (width_ci_95_low / width) * JND
    JND_ci_95_high = (width_ci_95_high / width) * JND

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
    df_group = combined_df.groupby(
        ["subject", "standard_angle_abs", "standard_center_frequency", "comparison_center_frequency", "trial_type"])
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
    PSE = round(PSE, 2)
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
                               JND_offset: float = .0,
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
        PSE = result.threshold(.5, return_ci=False)
        JND = result.threshold(.84, return_ci=False)
        # ax.vlines(x=JND, ymin=0, ymax=0.84, color=line_color, ls=line_style, lw=line_width)
        ax.vlines(x=JND, ymin=0, ymax=0.84, color="lightgrey", alpha=.3)
        ax.vlines(x=PSE, ymin=0, ymax=0.5, color="lightgrey", alpha=.3)
        rect = matplotlib.patches.Rectangle((PSE, JND_offset + 0.5), JND - PSE, 0.02, color=line_color)
        ax.add_patch(rect)
        # ax.hlines(y=JND_offset, xmin=PSE, xmax=JND, color=line_color, lw=6)

    # AXIS SETTINGS
    plt.axis('tight')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_ylim([ymin, 1])
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def reorderLegend(ax=None, order=None, unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None:  # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order, range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t, keys=keys: keys.get(t[0], np.inf)))
    if unique:
        labels, handles = zip(*unique_everseen(zip(labels, handles), key=labels))  # Keep only the first of each handle
    ax.legend(handles, labels)
    return handles, labels


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x, k in zip(seq, key) if not (k in seen or seen_add(k))]


def plot_pfs(subject, standard_center_frequency, comparison_center_frequency, plot_parameter=True, plot_JND=False):
    df = get_model_table(subject)
    df = df[(df.standard_center_frequency == standard_center_frequency)
            & (df.comparison_center_frequency == comparison_center_frequency)]
    if len(df) > 3:
        JND_TT = df.loc[df.trial_type == "ITD-->ITD", "JND"].values[0]
        JND_LL = df.loc[df.trial_type == "ILD-->ILD", "JND"].values[0]
        sigma_TT = JND_TT / np.sqrt(2)
        sigma_LL = JND_LL / np.sqrt(2)
        # JND predictions without a prior
        df.loc[df.trial_type == "ILD-->ITD", "JND_pred_no_prior"] = np.sqrt(sigma_LL ** 2 + sigma_TT ** 2)
        df.loc[df.trial_type == "ITD-->ILD", "JND_pred_no_prior"] = np.sqrt(sigma_TT ** 2 + sigma_LL ** 2)
        # JND predictions with IC model
        df.loc[df.trial_type == "ILD-->ITD", "JND_pred_IC"] = JND_TT
        df.loc[df.trial_type == "ITD-->ILD", "JND_pred_IC"] = JND_LL
    folder_path = DIR / "ps_results" / subject / f"{standard_center_frequency}-->{comparison_center_frequency}"
    ps_result_file_paths = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    cmap = matplotlib.cm.get_cmap('tab20')
    line_kws = {
        "ITD-->ITD": {"color": cmap(2), "ls": "-", "JND_offset": -0.06},
        "ILD-->ITD": {"color": cmap(3), "ls": "--", "JND_offset": -0.08},
        "ILD-->ILD": {"color": cmap(0), "ls": "-", "JND_offset": -0.02},
        "ITD-->ILD": {"color": cmap(1), "ls": "--", "JND_offset": -0.04}
    }
    fig, (ax_0, ax_1) = plt.subplots(1, 2, sharey=False, width_ratios=[3, 1], figsize=(16, 6))
    ax_0.axhline(y=1.0, color="lightgrey", alpha=.3)
    ax_0.axhline(y=0.5, color="lightgrey", alpha=.3)
    ax_0.axvline(x=0, color="lightgrey", alpha=.3)
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
                                         plot_data=not plot_JND,
                                         data_color=line_kws[trial_type]["color"],
                                         data_size=.5,
                                         extrapolate_stimulus=0,
                                         line_style=line_kws[trial_type]["ls"],
                                         line_color=line_kws[trial_type]["color"],
                                         line_width=3,
                                         label=trial_type,
                                         JND_offset=line_kws[trial_type]["JND_offset"],
                                         x_label="Stimulus displacement [°]",
                                         y_label="Proportion 'right'")
    ax_0.set_yticks([0.5, 0.84, 1.])
    ax_0.legend()
    trial_type_order = ["ILD-->ILD", "ITD-->ILD", "ILD-->ITD", "ITD-->ITD"]
    hue_order = ["ILD-->ILD", "ITD-->ILD", "ITD-->ITD", "ILD-->ITD"]
    reorderLegend(ax_0, trial_type_order)
    sns.barplot(df, ax=ax_1, x="trial_type", y="JND",
                hue="trial_type",
                hue_order=hue_order,
                order=trial_type_order,
                palette="tab20")
    if len(df) > 3:
        for bar, curr_trial_type in zip(ax_1.patches, hue_order):
            x = bar.get_x()
            width = bar.get_width()
            JND_pred_no_prior = df.loc[df.trial_type == curr_trial_type, "JND_pred_no_prior"].values[0]
            JND_ci_95_low = df.loc[df.trial_type == curr_trial_type, "JND_ci_95_low"].values[0]
            JND_ci_95_high = df.loc[df.trial_type == curr_trial_type, "JND_ci_95_high"].values[0]
            JND_pred_IC = df.loc[df.trial_type == curr_trial_type, "JND_pred_IC"].values[0]

            ax_1.hlines(y=JND_pred_no_prior, xmin=x, xmax=x + width, ls="--", color='tab:red', linewidth=2)
            ax_1.hlines(y=JND_pred_IC, xmin=x, xmax=x + width, ls="--", color='tab:green', linewidth=2)
            # Plot errorbars
            ax_1.vlines(x=x + width/2, ymin=JND_ci_95_low, ymax=JND_ci_95_high, color="black")
            ax_1.hlines(y=JND_ci_95_high, xmin=x + width/2 - 0.05, xmax=x + width/2 + 0.05, color="black")
            ax_1.hlines(y=JND_ci_95_low, xmin=x + width / 2 - 0.05, xmax=x + width / 2 + 0.05, color="black")

    ax_1.set_ylim(0, df["JND_ci_95_high"].max() * 1.1)
    ax_1.set_xlabel("Cue comparisons", fontsize=14)
    ax_1.set_ylabel("JND [°]", fontsize=14)
    ax_1.set_title("Discrimination thresholds", fontsize=14)
    line_1 = matplotlib.lines.Line2D([], [], color="tab:red", ls="--", label="SDT prediction")
    line_2 = matplotlib.lines.Line2D([], [], color="tab:green", ls="--", label="IC prediction")
    ax_1.legend(handles=[line_1, line_2], loc="upper left")

    title = f"{subject} at {standard_center_frequency}-->{comparison_center_frequency}Hz"
    fig.suptitle(title, fontsize=14)

