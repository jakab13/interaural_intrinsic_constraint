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

folder_path = DIR / "AV_depth_data" / "raw"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
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

combined_df["score"] = combined_df.apply(lambda row: int(row["is_correct"]) if row["stimulus_max_disparity"] > 15 else 1 - row["is_correct"], axis=1)


def get_psychometric_estimates(g):
    subject_id = g["subject_id"].unique()[0]
    data = g.groupby("stimulus_max_disparity", as_index=False).agg(
        {"score": "sum", g.columns[0]: "count"}).rename(
        columns={g.columns[0]: 'n_total'})
    res = ps.psignifit(
        data.values,
        sigmoid="norm",
        experiment_type="equal asymptote"
    )
    ps_folder_path = DIR / "AV_depth_data" / "ps_results"
    ps_result_file_path = ps_folder_path / Path(str(subject_id + ".json"))
    res.save_json(ps_result_file_path)


df_group = combined_df.groupby(["subject_id"])
df_model = df_group.apply(lambda g: get_psychometric_estimates(g), include_groups=True).reset_index()


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


def plot_pfs(save_fig=False):
    subject_ids = ["jakab_no_sound", "jakab_with_AM", "jakab_with_FM"]
    cmap = matplotlib.cm.get_cmap('tab20')
    c_dict = {
        "jakab_no_sound": cmap(0),
        "jakab_with_AM": cmap(2),
        "jakab_with_FM": cmap(4)
    }
    fig, ax= plt.subplots(figsize=(10, 6))
    for subject_id in subject_ids:
        ps_folder_path = DIR / "AV_depth_data" / "ps_results"
        ps_result_file_path = ps_folder_path / Path(str(subject_id + ".json"))
        res = ps.Result.load_json(ps_result_file_path)
        plot_psychometric_function_local(res,
                                         ax=ax,
                                         data_color=c_dict[subject_id],
                                         line_color=c_dict[subject_id],
                                         label=subject_id,
                                         data_size=.2,
                                         x_label="Depth")
    plt.legend()
    if save_fig:
        plt.savefig( DIR / "AV_depth_data" / "AV_depth_psychometrics.png", dpi=200)


plot_pfs(save_fig=True)
