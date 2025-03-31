import matplotlib
matplotlib.use('MacOSX')
import copy
import slab
import numpy as np
import datetime
import time
import data_handler
from sound_handler import apply_cue

subject = "jakab_matched_single"

# standard_angle_conditions = [-4, -3, -2, -1, 0, 0, 1, 2, 3, 4]
standard_angle_conditions = [0]
# comparison_angle_conditions = np.asarray([-25, -15, -10, -5, 5, 10, 15, 25])
comparison_angle_conditions = np.asarray([-15, -10, -5, 0, 5, 10, 15])
standard_cue = "ITD"
comparison_cue = "ITD"
stim_type = "noise_filtered_third_octave"
n_reps = 5
standard_center_frequency = comparison_center_frequency = 1200
ISI = 0.2
save = True
level = 70

df_trial_sequence = data_handler.generate_trial_sequence(standard_angle_conditions, comparison_angle_conditions, n_reps)
trial_counter = 0

columns = data_handler.get_columns()
results_table = slab.ResultsTable(columns=columns, subject=subject)

for seq_idx, seq_row in df_trial_sequence.iterrows():
    standard_angle = seq_row["standard_angle"]
    comparison_angle = seq_row["comparison_angle"]
    trial_type = standard_cue + "-->" + comparison_cue
    low_cutoff = standard_center_frequency / (2 ** (1 / 6))
    high_cutoff = standard_center_frequency * (2 ** (1 / 6))

    standard_stim = slab.Binaural.whitenoise(duration=0.3, samplerate=44100)
    # standard_stim_l = slab.Sound.whitenoise(duration=0.3, samplerate=44100)
    # standard_stim_r = slab.Sound.whitenoise(duration=0.3, samplerate=44100)
    # standard_stim = slab.Binaural([standard_stim_l, standard_stim_r])
    standard_stim = standard_stim.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
    standard_stim.level = level
    standard_stim = standard_stim.ramp(duration=0.01)

    # comparison_stim_l = slab.Sound.whitenoise(duration=0.3, samplerate=44100)
    # comparison_stim_r = slab.Sound.whitenoise(duration=0.3, samplerate=44100)
    # comparison_stim = slab.Binaural([comparison_stim_l, comparison_stim_r])
    # comparison_stim = comparison_stim.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
    # comparison_stim.level = level
    # comparison_stim = comparison_stim.ramp(duration=0.01)
    comparison_stim = copy.deepcopy(standard_stim)

    standard_stim = apply_cue(standard_stim, standard_cue, standard_angle, standard_center_frequency)
    comparison_stim = apply_cue(comparison_stim, comparison_cue, comparison_angle, comparison_center_frequency)
    presentations = [standard_stim, comparison_stim]
    order = np.random.permutation(len(presentations))
    datetime_onset = datetime.datetime.now()
    for i, idx in enumerate(order):
        current_stim = presentations[idx]
        current_stim.play()
        if i == 0:
            time.sleep(ISI)
    datetime_offset = datetime.datetime.now()
    while True:
        response = input(f"Which way did the sound move? <----- 1 | 2 -----> ({trial_counter + 1}/{len(df_trial_sequence)})")
        if response == "49" or response == "1":
            response = "left"
            break
        elif response == "51" or response == "2":
            response = "right"
            break
        else:
            continue
    reaction_time = datetime.datetime.now() - datetime_offset
    expected_standard = np.where(order == 0)[0][0]
    is_comparison_to_the_right = comparison_angle > standard_angle
    solution = None
    if expected_standard == 0 and is_comparison_to_the_right == 0:
        solution = "left"
    elif expected_standard == 0 and is_comparison_to_the_right == 1:
        solution = "right"
    elif expected_standard == 1 and is_comparison_to_the_right == 0:
        solution = "right"
    elif expected_standard == 1 and is_comparison_to_the_right == 1:
        solution = "left"
    is_correct = response == solution
    score = None
    if is_comparison_to_the_right == 1:
        score = int(is_correct)
    elif is_comparison_to_the_right == 0:
        score = 1 - int(is_correct)
    comparison_angle_abs = comparison_angle * -1 if standard_angle < 0 else comparison_angle

    if standard_angle < 0 and score == 0:
        score_abs = score + 1
    elif standard_angle < 0 and score == 1:
        score_abs = score - 1
    else:
        score_abs = score

    trial_counter += 1
    if save:
        row = results_table.Row(
            subject=subject,
            datetime_onset=datetime_onset,
            stim_type=stim_type,
            trial_type=trial_type,
            trial_index=seq_idx,
            standard_angle=standard_angle,
            standard_value="...",  # (ms or dB) subject specific based on HRTF
            standard_cue=standard_cue,
            standard_center_frequency=standard_center_frequency,
            comparison_angle=comparison_angle,
            comparison_value="...",  # (ms or dB) subject specific based on HRTF
            comparison_cue=comparison_cue,
            comparison_center_frequency=comparison_center_frequency,
            standard_order=np.where(order == 0)[0][0],
            comparison_order=np.where(order == 1)[0][0],
            solution=solution,
            inter_stimulus_interval=ISI,
            response=response,
            is_correct=is_correct,
            score=score,
            score_abs=score_abs,
            standard_angle_abs=abs(standard_angle),
            reaction_time=reaction_time,
            comparison_angle_abs=comparison_angle_abs
        )
        results_table.write(row)
    time.sleep(.5)

