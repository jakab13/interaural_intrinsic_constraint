import slab
import time
import os
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import scipy
import datetime
from mplcursors import cursor
import copy
import matplotlib.pyplot as plt
import random
import psignifit as ps

# ==============================================================

output_path = "interaural_intrinsic_constraint_23.csv"
subject_id = "jakab"
standard_cue = "ild"  # itd or ild or both
comparison_cue = "ild"  # itd or ild or both
stim_type = "broadband"  # "broadband" or "hp" or "lp" or "lp_200"
centre_frequency = 500
standard_angle = 2  # [0, 2, 4, 6, 8]
save = True
n_reps = 2

# ==============================================================

def make_interaural_level_spectrum():
    hrtf = slab.HRTF.kemar()  # load KEMAR by default
    # get the filters for the frontal horizontal arc
    idx = np.where((hrtf.sources.vertical_polar[:, 1] == 0) & (
        (hrtf.sources.vertical_polar[:, 0] <= 90) | (hrtf.sources.vertical_polar[:, 0] >= 270)))[0]
    # at this point, we could just get the transfer function of each filter with hrtf.data[idx[i]].tf(),
    # but it may be better to get the spectral left/right differences with ERB-spaced frequency resolution:
    azi = hrtf.sources.vertical_polar[idx, 0]
    # 270<azi<360 -> azi-360 to get negative angles on the left
    azi[azi >= 270] = azi[azi >= 270]-360
    sort = np.argsort(azi)
    fbank = slab.Filter.cos_filterbank(samplerate=hrtf.samplerate, pass_bands=True)
    freqs = fbank.filter_bank_center_freqs()
    noise = slab.Sound.pinknoise(duration=5., samplerate=hrtf.samplerate)
    noise_0 = slab.Binaural(hrtf.data[idx[0]].apply(noise))
    noise_0_bank = fbank.apply(noise_0.left)
    ils = dict()
    ils['samplerate'] = hrtf.samplerate
    ils['frequencies'] = freqs
    ils['azimuths'] = azi[sort]
    ils['level_diffs_right'] = np.zeros((len(freqs), len(idx)))
    ils['level_diffs_left'] = np.zeros((len(freqs), len(idx)))
    for n, i in enumerate(idx[sort]):  # put the level differences in order of increasing angle
        noise_filt = slab.Binaural(hrtf.data[i].apply(noise))
        noise_bank_left = fbank.apply(noise_filt.left)
        noise_bank_right = fbank.apply(noise_filt.right)
        ils['level_diffs_right'][:, n] = noise_0_bank.level - noise_bank_right.level
        ils['level_diffs_left'][:, n] = noise_0_bank.level - noise_bank_left.level
    return ils
#
#
# ils = make_interaural_level_spectrum()


def azimuth_to_ild(azimuth, frequency=2000, ils=None):
    level_diffs_left = ils['level_diffs_left']
    level_diffs_right = ils['level_diffs_right']
    # levels = [np.interp(azimuth, ils['azimuths'], level_diffs[i, :]) for i in range(level_diffs.shape[0])]
    levels_right = [np.interp(azimuth, ils['azimuths'], level_diffs_right[i, :]) for i in
                    range(level_diffs_right.shape[0])]
    levels_left = [np.interp(azimuth, ils['azimuths'], level_diffs_left[i, :]) for i in
                    range(level_diffs_left.shape[0])]
    ild_right = np.interp(frequency, ils['frequencies'], levels_right) * -1
    ild_left = np.interp(frequency, ils['frequencies'], levels_left) * -1
    return [ild_right, ild_left]  # interpolate level difference at frequency


ils = pickle.load(open('ils_jakab.pickle', 'rb'))
isi = 0.2
stim_name = stim_type + "_" + str(centre_frequency)

df = pd.DataFrame(columns=[
    "subject_id",
    "datetime_onset",
    "stim_type",
    "filter_frequency",
    "stim_name",
    "standard_cue",
    "comparison_cue",
    "trial_type",
    "standard_angle",
    "comparison_angle",
    "standard_order",
    "comparison_order",
    "isi",
    "response",
    "reaction_time",
    "is_correct",
    "score"
])

comparison_angle_conditions = [-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65]
trials = slab.Trialsequence(conditions=comparison_angle_conditions, n_reps=n_reps)
# trials = slab.Staircase(start_val=45, step_sizes=[5, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2], n_down=2, n_up=2, n_reversals=12)
# trials = slab.Trialsequence(conditions=[-45, -30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30, 45], n_reps=n_reps)

df_trial_sequence = pd.DataFrame()

# for standard_cue in ["itd", "ild"]:
#     for comparison_cue in ["itd", "ild"]:
#         trial_obj = {
#             "standard_cue": standard_cue,
#             "comparison_cue": comparison_cue,
#             "comparison_angle": slab.Trialsequence(conditions=comparison_angle_conditions, n_reps=n_reps)
#         }
#         df_curr_trial_obj = pd.DataFrame.from_dict(trial_obj)
#         df_trial_sequence = pd.concat([df_trial_sequence, df_curr_trial_obj])

# trial_obj = {
#     "standard_cue": "both",
#     "comparison_cue": "both",
#     "comparison_angle": slab.Trialsequence(conditions=comparison_angle_conditions, n_reps=n_reps)
# }
# df_curr_trial_obj = pd.DataFrame.from_dict(trial_obj)
# df_trial_sequence = pd.concat([df_trial_sequence, df_curr_trial_obj])

for standard_angle in [0, 2, 4, 6, 8]:
    trial_obj = {
        "standard_angle": standard_angle,
        "standard_cue": standard_cue,
        "comparison_cue": comparison_cue,
        "comparison_angle": slab.Trialsequence(conditions=comparison_angle_conditions, n_reps=n_reps)
    }
    df_curr_trial_obj = pd.DataFrame.from_dict(trial_obj)
    df_trial_sequence = pd.concat([df_trial_sequence, df_curr_trial_obj])


def apply_cue(stim, cue, angle, filter_frequency, head_radius=7.5):
    sound = copy.deepcopy(stim)
    if cue == "itd" or cue == "both":
        itd_val = slab.Binaural.azimuth_to_itd(angle, frequency=filter_frequency, head_radius=head_radius)
        sound = sound.itd(itd_val)
    elif cue == "ild" or cue == "both":
        # ild_val = slab.Binaural.azimuth_to_ild(angle, frequency=filter_frequency, ils=ils)
        # sound = sound.ild(ild_val).externalize()
        ild_vals = azimuth_to_ild(angle, frequency=filter_frequency, ils=ils)
        sound.level += ild_vals
    sound = sound.externalize()
    sound = sound.ramp(duration=0.05)
    return sound


trial_counter = 0
n_trials = trials.n_trials if hasattr(trials, 'n_trials') else None

df_trial_sequence = df_trial_sequence.sample(frac=1).reset_index(drop=True)

for seq_idx, seq_row in df_trial_sequence.iterrows():
    # standard_angle = random.choice([0, 2, 4, 6, 8])
    standard_angle = seq_row["standard_angle"]
    standard_cue = seq_row["standard_cue"]
    comparison_cue = seq_row["comparison_cue"]
    comparison_angle = seq_row["comparison_angle"]
    # noise = slab.Binaural.pinknoise(duration=0.3, samplerate=44100).ramp(duration=0.05)
    uso_ID = random.choice([2, 3, 5, 7, 10, 11, 14, 17, 18, 19, 20, 21, 22, 25, 27, 28])
    uso = slab.Binaural(f"/Users/jakabpilaszanovich/Documents/GitHub/distance_law_iacc_manipulation/tools/Audio/dry_USOs/N_uso_300ms_{uso_ID}_control.wav")
    uso.data = uso.data[: int(0.3 * uso.samplerate)]
    # lp_noise = slab.Binaural.pinknoise(duration=0.3, samplerate=44100).filter(1000, kind="lp").ramp(duration=0.05)
    noise = slab.Binaural.silence(duration=0.3, samplerate=44100)
    tone = slab.Binaural.tone(frequency=centre_frequency, duration=0.3, samplerate=44100).ramp(duration=0.05)
    noise.data = tone.data
    if stim_type != "broadband":
        filt = slab.Filter.band(frequency=centre_frequency, kind=stim_type, samplerate=44100)
        noise = filt.apply(noise)
        noise = noise.ramp(when="offset", duration=0.09)
    # direction = 1 if random.random() < 0.5 else -1
    # standard_angle *= direction
    standard = apply_cue(noise, standard_cue, standard_angle, centre_frequency)
    comparison = apply_cue(noise, comparison_cue, comparison_angle, centre_frequency)
    presentations = [standard, comparison]
    order = np.random.permutation(len(presentations))
    expected_standard = np.where(order == 0)[0][0]
    is_comparison_to_the_right = comparison_angle > standard_angle
    solution = None
    if expected_standard == 0 and is_comparison_to_the_right == 0:
        solution = 1
    elif expected_standard == 0 and is_comparison_to_the_right == 1:
        solution = 2
    elif expected_standard == 1 and is_comparison_to_the_right == 0:
        solution = 2
    elif expected_standard == 1 and is_comparison_to_the_right == 1:
        solution = 1
    datetime_onset = datetime.datetime.now()
    for i, idx in enumerate(order):
        stim = presentations[idx]
        stim.play()
        if i == 0:
            time.sleep(isi)
    datetime_offset = datetime.datetime.now()
    while True:
        response = input(f"Which way did the sound move? <----- 1 | 2 -----> ({trial_counter + 1}/{len(df_trial_sequence)})")
        if response not in ["1", "2", "49", "51"]:
            print(f"You entered: '{response}' response. You have to select <----- 1 | 2 ----->")
            continue
        else:
            break
    if response == "49" or response == "1":
        response = 1
    elif response == "51" or response == "2":
        response = 2
    reaction_time = datetime.datetime.now() - datetime_offset
    is_correct = response == solution
    score = int(is_correct) if is_comparison_to_the_right else 1 - int(is_correct)
    # trials.add_response(score)
    # if hasattr(trials, "plot"):
    #     trials.plot()
    trial_counter += 1
    if save:
        row = {
            "subject_id": subject_id,
            "datetime_onset": datetime_onset,
            "stim_type": stim_type,
            "filter_frequency": centre_frequency,
            "stim_name": stim_name,
            "standard_cue": standard_cue,
            "comparison_cue": comparison_cue,
            "trial_type": standard_cue + "-->" + comparison_cue,
            "standard_angle": standard_angle,
            "comparison_angle": comparison_angle,
            "standard_order": np.where(order == 0)[0][0],
            "comparison_order": np.where(order == 1)[0][0],
            "isi": isi,
            "response": response,
            "reaction_time": reaction_time,
            "is_correct": is_correct,
            "score": score
        }
        df_curr = pd.DataFrame.from_dict([row])
        df_curr.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


# =============== PLOTTING ==================

df_final = pd.read_csv(output_path)
# df_final = df_final[~(df_final.trial_type == "both-->both")]
# df_final = df_final[~(df_final.trial_type == "itd-->both")]
# df_final = df_final[~(df_final.trial_type == "ild-->both")]
# df_final = df_final[~(df_final.subject_id == "jakab2")]
# df_final = df_final.replace("jakab2", "jakab")


def get_psychometric_mu(df_group):
    options = {"sigmoidName": "norm", "expType": "equalAsymptote", "useGPU": 1}
    data = df_group.groupby("comparison_angle", as_index=False).agg(
        {"score": "sum", df_group.columns[0]: "count"}).rename(
        columns={df_group.columns[0]: 'n_total'})
    res = ps.psignifit(data.values, options)
    mu = res["Fit"][0]
    return mu


def get_psychometric_sigma(df_group):
    options = {"sigmoidName": "norm", "expType": "equalAsymptote", "useGPU": 1}
    data = df_group.groupby("comparison_angle", as_index=False).agg(
        {"score": "sum", df_group.columns[0]: "count"}).rename(
        columns={df_group.columns[0]: 'n_total'})
    res = ps.psignifit(data.values, options)
    sigma = res["Fit"][1]
    return sigma


def get_sigma(row):
    cdf = scipy.stats.norm.cdf
    mean, sigma = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100], maxfev=1000)[0]
    return sigma


def get_mu(row):
    cdf = scipy.stats.norm.cdf
    mean, sigma = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100], maxfev=1000)[0]
    return mean

def get_mu_err(row):
    cdf = scipy.stats.norm.cdf
    [popt, pcov] = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100], maxfev=1000)
    perr = np.sqrt(np.diag(pcov))
    return perr[0]

def get_sigma_err(row):
    cdf = scipy.stats.norm.cdf
    [popt, pcov] = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100], maxfev=1000)
    perr = np.sqrt(np.diag(pcov))
    return perr[1]


df_group = df_final.groupby(["subject_id", "standard_angle", "stim_name", "trial_type"])
df_model = df_group.apply(lambda g: get_psychometric_mu(g)).reset_index(name='mu')
df_model["sigma"] = df_group.apply(lambda g: get_psychometric_sigma(g)).reset_index(name='sigma')["sigma"]
# df_model = df_group.apply(lambda g: get_mu(g)).reset_index(name='mu')
# df_model["sigma"] = df_group.apply(lambda g: get_sigma(g)).reset_index(name='sigma')["sigma"]
df_model["mu_sd"] = df_group.apply(lambda g: get_mu_err(g)).reset_index(name='mu_sd')["mu_sd"]
df_model["sigma_sd"] = df_group.apply(lambda g: get_sigma_err(g)).reset_index(name='sigma_sd')["sigma_sd"]


df_final = df_final.merge(df_model[["subject_id", "standard_angle", "stim_name", "trial_type", "mu", "sigma", "mu_sd", "sigma_sd"]],
                          on=["subject_id", "standard_angle", "stim_name", "trial_type"])


def get_cue_combination(row):
    if row["standard_cue"] == row["comparison_cue"]:
        cue_combination = "within"
    else:
        cue_combination = "across"
    if (row["standard_cue"] == "both") or (row["comparison_cue"] == "both"):
        cue_combination = "combined"
    if (row["standard_cue"] == "both") and (row["comparison_cue"] == "both"):
        cue_combination = "within"
    return cue_combination

df_final["cue_combination"] = df_final.apply(lambda row: get_cue_combination(row), axis=1)


df_cue_ratio = df_final.groupby(["subject_id",
                                 "stim_name",
                                 "standard_angle",
                                 "trial_type"
                                 ], as_index=False)["mu", "sigma"].mean()

df_final["sigma_itd"] = None
df_final["sigma_ild"] = None

for subject_id in df_final.subject_id.unique():
    for stim_name in df_final.stim_name.unique():
        for standard_angle in df_final.standard_angle.unique():
            for trial_type in df_final.trial_type.unique():
                sigma_itd = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                         (df_cue_ratio.stim_name == stim_name) &
                                         (df_cue_ratio.standard_angle == standard_angle) &
                                         (df_cue_ratio.trial_type == "itd-->itd")]["sigma"]
                sigma_ild = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                         (df_cue_ratio.stim_name == stim_name) &
                                         (df_cue_ratio.standard_angle == standard_angle) &
                                         (df_cue_ratio.trial_type == "ild-->ild")]["sigma"]
                mu_itd = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                      (df_cue_ratio.stim_name == stim_name) &
                                      (df_cue_ratio.standard_angle == standard_angle) &
                                      (df_cue_ratio.trial_type == "itd-->itd")]["mu"]
                mu_ild = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                      (df_cue_ratio.stim_name == stim_name) &
                                      (df_cue_ratio.standard_angle == standard_angle) &
                                      (df_cue_ratio.trial_type == "ild-->ild")]["mu"]
                if len(sigma_itd):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.stim_name == stim_name) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type), "sigma_itd"] = sigma_itd.values[0].astype(float)
                if len(sigma_ild):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.stim_name == stim_name) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type), "sigma_ild"] = sigma_ild.values[0].astype(float)
                if len(mu_itd):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.stim_name == stim_name) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type), "mu_itd"] = mu_itd.values[0].astype(float)
                if len(mu_ild):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.stim_name == stim_name) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type), "mu_ild"] = mu_ild.values[0].astype(float)

# df_final = df_final.fillna(0)

#
# def get_PSE_mid(row):
#     x_0 = row["standard_angle"]
#     sigma_i = "sigma_" + str(row["standard_cue"])
#     sigma_j = "sigma_" + str(row["comparison_cue"])
#     mu_i = "mu_" + str(row["standard_cue"])
#     mu_j = "mu_" + str(row["comparison_cue"])
#     sigma_i = row[sigma_i]
#     sigma_j = row[sigma_j]
#     mu_i = row[mu_i]
#     mu_j = row[mu_j]
#     sin_theta_i = 1 / np.sqrt(1 + sigma_i ** 2)
#     sin_theta_j = 1 / np.sqrt(1 + sigma_j ** 2)
#     # Q = np.sqrt(1 + sigma_j ** 2) / np.sqrt(1 + sigma_i ** 2)
#     # PSE = x_0 - mu_j + mu_i * (sin_theta_j / sin_theta_i)
#     # PSE = x_0 * sin_theta_i / sin_theta_j
#     PSE = None
#     if row["standard_cue"] == row["comparison_cue"]:
#         PSE = mu_i
#     if sigma_i == 0 or sigma_j == 0:
#         PSE = None
#     else:
#         # if sigma_j > sigma_i:
#         PSE = x_0 * sigma_j / sigma_i + mu_i
#         # elif sigma_j < sigma_i:
#         #     PSE = x_0 * sigma_i / sigma_j + mu_j
#     return PSE
#
#
# # df_final["PSE_base"] = df_final.apply(lambda row: get_PSE(row), axis=1)
#
# df_final["PSE_mid"] = df_final.apply(lambda row: get_PSE_mid(row), axis=1)
# df_final["PSE_mid_y"] = 0.5
# df_final["PSE_high"] = df_final.apply(lambda row: get_PSE_high(row), axis=1)
# df_final["PSE_high_y"] = 0.85
# df_final["PSE_low"] = df_final.apply(lambda row: get_PSE_low(row), axis=1)

# df_final["delta_x"] =
df_final["delta_mu"] = df_final["mu"] - df_final["standard_angle"]
df_final["delta_mu_min"] = df_final["delta_mu"] - df_final["mu_sd"]
df_final["delta_mu_max"] = df_final["delta_mu"] + df_final["mu_sd"]


def get_pred_bayes(row):
    if row["cue_combination"] != "combined":
        sigma_i = row["sigma_" + str(row["standard_cue"])]
        sigma_j = row["sigma_" + str(row["comparison_cue"])]
        w_i = sigma_j ** 2 / (sigma_j ** 2 + sigma_i ** 2)
        w_j = sigma_i ** 2 / (sigma_j ** 2 + sigma_i ** 2)
        standard_pred = w_i * row["standard_angle"] + w_j * row["standard_angle"]
        comparison_pred = w_i * row["comparison_angle"] + w_j * row["comparison_angle"]
        delta_pred = standard_pred - comparison_pred
        return delta_pred


def get_pred_IC(row):
    if row["cue_combination"] != "combined":
        sigma_i = row["sigma_" + str(row["standard_cue"])]
        sigma_j = row["sigma_" + str(row["comparison_cue"])]
        mu_i = row["mu_" + str(row["standard_cue"])]
        mu_j = row["mu_" + str(row["comparison_cue"])]
        sin_i = (1 / np.sqrt(1 + (sigma_i ** 2)))
        sin_j = (1 / np.sqrt(1 + (sigma_j ** 2)))
        direction = 1 if (sigma_i < sigma_j and row["cue_combination"] == "across") else -1
        # delta_pred = Q * (row["standard_angle"] - row["comparison_angle"]) * direction
        # delta_pred = sin_i / sin_j * (row["standard_angle"] - row["comparison_angle"]) * direction
        # delta_pred = (sigma_j / sigma_i) * (row["standard_angle"] - row["comparison_angle"]) * direction
        # delta_pred = Q * (row["standard_angle"]) + (1 / Q * (row["comparison_angle"]) * direction)
        delta_pred = sigma_j * row["standard_angle"] - sigma_i * row["comparison_angle"]
        return delta_pred


# df_final["pred_bayes"] = df_final.apply(lambda row: get_pred_bayes(row), axis=1)
# df_final["pred_IC"] = df_final.apply(lambda row: get_pred_IC(row), axis=1)
#
# df_final = df_final.merge(df_final.groupby(["subject_id", "stim_name", "standard_angle", "trial_type"],
#                                            as_index=False)["pred_bayes", "pred_IC"].mean(),
#                           on=["subject_id", "stim_name", "standard_angle", "trial_type"],
#                           suffixes=("", "_mean"))

df_plot = df_final
    # (df_final["stim_name"] == "lp_1500")
    # & (df_final["standard_angle"] == 4)



g = sns.FacetGrid(df_plot, hue="trial_type",
                  col="cue_combination",
                  row="standard_angle",
                  col_order=[
                      "within",
                      "across",
                      # "combined"
                  ],
                  hue_order=[
                      "itd-->itd",
                      "ild-->ild",
                      # "both-->both",
                      "ild-->itd",
                      "itd-->ild",
                      # "ild-->both",
                      # "itd-->both"
                  ],
                  height=2, aspect=2, sharex=True, sharey=True)
g.refline(y=0.5, c="grey", alpha=.3)
[ax.axvline(float(ax.title.get_text().split("standard_angle = ")[1].split("|")[0]), ls='--', c="black", alpha=.3) for ax in g.axes.flatten()]
g.set(xlim=(-45, 45))
g.map(sns.regplot, "comparison_angle", "score", logistic=True, scatter=False, ci=None)
# g.map(sns.lineplot, "comparison_angle", "score")
def plot_PSE(x, **kwargs):
    c = kwargs.get("color", "g")
    x_mean = x.mean()
    plt.axvline(x_mean, ymax=0.5, c=c, alpha=.5)
    plt.plot(x_mean, 0.5, "o", c=c, markersize=10)
g.map(plot_PSE, "mu")
g.add_legend()
# g.set_titles(template="standard stim at: {row_name}°")
g.set_titles(template="standard at: {row_name}")
g.figure.suptitle("PSE of 'comparison stimuli' in different cue combinations", size=16)
g.figure.subplots_adjust(top=.9)
g.set_xlabels(label="comparison stim azimuth")
# plt.savefig("Interaural Intrinsic Constraint pilot PSEs (1-16 deg).png", dpi=400)
# cursor(hover=True)

# df_itd_vs_ild = df_plot.groupby(["subject_id", "trial_type", "stim_name", "standard_angle"], as_index=False)["mu", "sigma"].mean().pivot(
#     index=["subject_id", "stim_name", "standard_angle"], columns=["trial_type"])
# df_itd_vs_ild.columns = ['_'.join(tup).rstrip('_') for tup in df_itd_vs_ild.columns.values]
# df_itd_vs_ild = df_itd_vs_ild.reset_index()
#
# df_itd_vs_ild = df_itd_vs_ild.merge(df_itd_vs_ild.groupby(["subject_id"],
#                                           as_index=False)["sigma_itd-->ild", "sigma_ild-->itd"].mean(),
#                     on=["subject_id"], suffixes=("", "_mean"))
#
# df_itd_vs_ild["mu_ratio"] = abs(df_itd_vs_ild["mu_itd-->ild"]) / abs(df_itd_vs_ild["mu_ild-->itd"])
# df_itd_vs_ild["mu_pred"] = df_itd_vs_ild["mu_ratio"] * df_itd_vs_ild["standard_angle"]
# df_itd_vs_ild["sigma_ratio"] = abs(df_itd_vs_ild["sigma_ild-->itd"]) / abs(df_itd_vs_ild["sigma_itd-->ild"])
# df_itd_vs_ild["sigma_pred"] = df_itd_vs_ild["sigma_ratio"] * df_itd_vs_ild["standard_angle"]

g_2 = sns.FacetGrid(df_plot,
                    hue="trial_type",
                    hue_order=["itd-->itd", "ild-->ild",
                               # "both-->both",
                               "ild-->itd", "itd-->ild",
                               # "ild-->both", "itd-->both", "both-->ild", "both-->itd"
                               ],
                    col="cue_combination",
                    col_order=["within", "across",
                               # "combined"
                               ],
                    # row="stim_name",
                    # row_order=["broadband_2000", "lp_500", "lp_1000"],
                    height=3)
g_2.refline(x=0, c="grey", alpha=.2)
g_2.refline(y=0, c="grey", alpha=.2)
# g_2.map(sns.lineplot, "standard_angle", "pred_bayes_mean", ls="dotted")
# g_2.map(sns.scatterplot, "standard_angle", "pred_IC_mean", marker=(4, 2, 0))
# g_2.map(sns.lineplot, "standard_angle", "pred_IC_mean_mult", ls="dashed", alpha=.2)
g_2.map(sns.lineplot, "standard_angle", "delta_mu")
g_2.map(sns.scatterplot, "standard_angle", "delta_mu", label=False)
# g_2.map(sns.lineplot, "standard_angle", "delta_mu_min", alpha=.01)
# g_2.map(sns.lineplot, "standard_angle", "delta_mu_max", alpha=.01)
# g_2.set(ylim=(-15, 40))
g_2.add_legend()
# [ax.legend() for ax in g_2.axes.flatten()]


# ======== PSE error plot

# g_pse_error = sns.FacetGrid(df_plot,
#                             hue="trial_type",
#                             hue_order=["itd-->itd", "ild-->ild", "both-->both", "ild-->itd", "itd-->ild", "ild-->both", "itd-->both"],
#                             col="cue_combination",
#                             col_order=["within", "across", "combined"],
#                             # row="stim_name",
#                             height=3)
# g_pse_error.map(sns.lineplot, "standard_angle", "sigma")
# g_pse_error.map(sns.scatterplot, "standard_angle", "sigma")
# g_pse_error.add_legend()


# g_3 = sns.FacetGrid(
#     df_itd_vs_ild,
#     row="stim_name"
# )
# g_3.refline(x=0, c="grey", alpha=.2)
# g_3.refline(y=0, c="grey", alpha=.2)
# g_3.map(sns.regplot, "standard_angle", "sigma_ratio", color="black", ci=None)
# # g_3.map(sns.regplot, "standard_angle", "sigma_pred", color="grey", scatter=False)
# g_3.add_legend()

# g_sigma_ratio = sns.FacetGrid(df_plot)
# g_sigma_ratio.map(sns.regplot, "sigma_itd", "sigma_ild")

df_count_check = df_final.groupby(["subject_id", "stim_name", "standard_angle", "trial_type", "comparison_angle"]).count()
print(df_count_check[df_count_check["Unnamed: 0"] < 300]["Unnamed: 0"].to_string())



# stairs = slab.Staircase(start_val=45, n_reversals=18, step_sizes=[5, 4, 3, 2, 2, 1])
# standard_cue = "ild"
# comparison_cue = "itd"
# standard_angle = 3
# centre_frequency = 2000
# stim_type = "lp"

# for comparison_angle in stairs:
#     noise = slab.Binaural.pinknoise(duration=0.3, samplerate=44100).ramp(duration=0.05)
#     if stim_type != "broadband":
#         filt = slab.Filter.band(frequency=centre_frequency, kind=stim_type, samplerate=44100)
#         noise = filt.apply(noise)
#         noise = noise.ramp(when="offset", duration=0.09)
#     standard = apply_cue(noise, standard_cue, standard_angle, centre_frequency)
#     comparison = apply_cue(noise, comparison_cue, comparison_angle, centre_frequency)
#     presentations = [standard, comparison]  # assuming two sound objects
#     order = np.random.permutation(2)
#     for idx in order:
#         stim = presentations[idx]
#         stim.play()
#         time.sleep(0.1)
#     while True:
#         response = input(f"Which way did the sound move? <----- 1 | 2 ----->")
#         if response not in ["1", "2", "49", "51"]:
#             print(f"You entered: '{response}' response. You have to select <----- 1 | 2 ----->")
#             continue
#         else:
#             break
#     if response == "49" or response == "1":
#         response = 1
#     elif response == "51" or response == "2":
#         response = 2
#     interval = np.where(order == 0)[0][0]
#     response = response == (interval + 1)
#     print(not response)
#     stairs.add_response(int(not response))  # initiates calculation of next stimulus value
#     stairs.plot()
# print(stairs.threshold())


