from experiment import Experiment
from experiment_analysis import *

subject = "pp00"

exp = Experiment(subject)

exp.standard_center_frequency = 1000
exp.comparison_center_frequency = 1000

exp.head_radius = 8.0  # in cm
reference_angle = 10
PSE_estimate = 10  # initial guess of PSE

exp.standard_cue = "ILD"
exp.comparison_cue = "ILD"
exp.standard_angle = reference_angle
exp.PSE_angle = reference_angle

# Familiarisation (without saving data)
# exp.n_reps = 2
# exp.run_sequence(save=False)

# Number of repetitions for one sequence
exp.n_reps = 4  # should be an even number

# ===========================================================================
exp.mixing_gain = 0.5
exp.run_sequence()

# ===========================================================================
exp.mixing_gain = 0.6
exp.run_sequence()

# ===========================================================================
exp.mixing_gain = 0.7
exp.run_sequence()

# ===========================================================================
exp.mixing_gain = 0.8
exp.run_sequence()

# ===========================================================================
exp.mixing_gain = 0.9
exp.run_sequence()

# ===========================================================================
exp.mixing_gain = 1
exp.run_sequence()


# ===========================================================================
# Plotting ==================================================================
subject = "pp04"
df = get_df(subject, exp.standard_center_frequency, exp.comparison_center_frequency)
sns.lmplot(df, x="comparison_angle_abs", y="score_abs", hue="mixing_gain", ci=None, logistic=True, scatter=False, palette="copper")
# sns.lineplot(df, x="comparison_angle_abs", y="score_abs", hue="mixing_gain", marker="o", errorbar=None, palette="copper")
df_groups = df.groupby("mixing_gain")

for g_name, g in df_groups:
    data = g.groupby("comparison_angle_abs", as_index=False).agg(
            {"score_abs": "sum", g.columns[0]: "count"}).rename(
            columns={g.columns[0]: 'n_total'})
    res = ps.psignifit(
        data.values,
        sigmoid="norm",
        experiment_type="equal asymptote"
    )
    res.save_json(subject + "_MG-" + str(int(g_name * 10)) + ".json")

mg_vals = [0.5, 0.75, 1]

fig, axs = plt.subplots(5, 1)
for sub_idx, subject in enumerate(["pp00", "pp01", "pp02", "pp03", "pp04"]):
    filenames = [f"{subject}_MG-{i}.json" for i in [5, 7, 10]]
    results = [ps.Result.load_json(DIR / f) for f in filenames]
    palette = sns.color_palette("copper", n_colors=len(results))
    ax = axs[sub_idx]
    ax.set_title(subject)
    for idx, res in enumerate(results):
        psp.plot_psychometric_function(res, ax=ax, plot_data=False, line_color=palette[idx])
    ax.vlines(10, 0, 1, colors="red", linestyles="--", alpha=.3, label="standard stim")
    if sub_idx == 1:
        ax.legend()
    ax.set_xlim([-25, 40])
plt.xlim([-25, 40])

# =========
lower_bound = [res.confidence_intervals["threshold"]["0.95"][0] for res in results]
upper_bound = [res.confidence_intervals["threshold"]["0.95"][1] for res in results]
PSEs = [res.parameters_estimate_mean["threshold"] for res in results]
sns.lineplot(x=[5, 6, 7, 8, 9, 10], y=PSEs)
plt.fill_between([5, 6, 7, 8, 9, 10], lower_bound, upper_bound, alpha=.1)

# =========
lower_bound = [res.confidence_intervals["width"]["0.95"][0]/2 for res in results]
upper_bound = [res.confidence_intervals["width"]["0.95"][1]/2 for res in results]
PSEs = [res.parameters_estimate_mean["width"]/2 for res in results]
sns.lineplot(x=[5, 6, 7, 8, 9, 10], y=PSEs)
plt.fill_between([5, 6, 7, 8, 9, 10], lower_bound, upper_bound, alpha=.1)
