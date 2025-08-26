import pandas as pd
import numpy as np


def generate_trial_sequence(
        standard_angle,
        PSE_angle,
        standard_cue,
        comparison_cue,
        n_reps):
    comparison_angle_conditions = list()
    if comparison_cue == "ITD":
        comparison_angle_conditions = np.asarray([-20, -15, -10, -5, -5, 0, 0, 5, 5, 10, 15, 20])
    elif comparison_cue == "ILD":
        comparison_angle_conditions = np.asarray([-20, -15, -10, -5, -5, 0, 0, 5, 5, 10, 15, 20])
    elif comparison_cue == "BOTH":
        comparison_angle_conditions = np.asarray([-10, -6, -4, -2, -2, 0, 0, 2, 2, 4, 6, 10])
    comparison_angle_conditions = comparison_angle_conditions + PSE_angle
    df_trial_sequence = pd.DataFrame()
    for comparison_angle in comparison_angle_conditions:
        for n in range(n_reps):
            trial = {
                "standard_angle": standard_angle if n < n_reps/2 else -standard_angle,
                "comparison_angle": comparison_angle if n < n_reps/2 else -comparison_angle,
                "standard_cue": standard_cue,
                "comparison_cue": comparison_cue
            }
            df_trial = pd.DataFrame.from_dict([trial])
            df_trial_sequence = pd.concat([df_trial_sequence, df_trial])
    return df_trial_sequence


def get_columns():
    column_names = [
        "subject",
        "head_radius",
        "datetime_onset",
        "stim_type",
        "trial_type",
        "mixing_gain",
        "trial_index",
        "standard_angle",
        "standard_value",
        "standard_cue",
        "standard_center_frequency",
        "comparison_angle",
        "comparison_value",
        "comparison_cue",
        "comparison_center_frequency",
        "standard_order",
        "comparison_order",
        "solution",
        "inter_stimulus_interval",
        "response",
        "is_correct",
        "score",
        'score_abs',
        "standard_angle_abs",
        "reaction_time",
        'comparison_angle_abs'
    ]
    return column_names

