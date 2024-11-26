import pandas as pd


def generate_trial_sequence(standard_angle_conditions, comparison_angle_conditions, n_reps, randomise=True):
    df_trial_sequence = pd.DataFrame()
    for standard_angle in standard_angle_conditions:
        for comparison_angle in comparison_angle_conditions:
            for n in range(n_reps):
                trial = {
                    "standard_angle": standard_angle,
                    "comparison_angle": comparison_angle
                }
                df_trial = pd.DataFrame.from_dict([trial])
                df_trial_sequence = pd.concat([df_trial_sequence, df_trial])
    if randomise:
        df_trial_sequence = df_trial_sequence.sample(frac=1).reset_index(drop=True)
    return df_trial_sequence


def get_columns():
    column_names = [
        "subject",
        "datetime_onset",
        "stim_type",
        "trial_type",
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
        "standard_angle_abs",
        "reaction_time"
    ]
    return column_names

