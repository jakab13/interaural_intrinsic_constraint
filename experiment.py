import matplotlib
matplotlib.use('MacOSX')
import copy
import slab
import numpy as np
import datetime
import time
from data_handler import generate_trial_sequence, get_columns
from sound_handler import generate_stim, apply_cue
from utils import get_response


class Experiment:
    def __init__(self, subject):
        self.subject = subject
        self.standard_center_frequency = 1000
        self.comparison_center_frequency = 1000
        self.head_radius = 8  # in cm
        self.n_reps = 2

        self.standard_angle = 10
        self.PSE_angle = 5
        self.standard_cue = "ITD"
        self.comparison_cue = "ILD"

        self.ISI = 0.2
        self.stim_type = "noise_filtered_third_octave"
        self.columns = get_columns()

    def run_sequence(self, save=True):
        trial_seq = generate_trial_sequence(
            self.standard_angle,
            self.PSE_angle,
            self.standard_cue,
            self.comparison_cue,
            self.n_reps
        )
        trial_seq = trial_seq.sample(frac=1).reset_index(drop=True)
        if save:
            results_table = slab.ResultsTable(columns=self.columns, subject=self.subject)
        for seq_idx, seq_row in trial_seq.iterrows():
            standard_angle = seq_row["standard_angle"]
            comparison_angle = seq_row["comparison_angle"]
            standard_cue = seq_row["standard_cue"]
            comparison_cue = seq_row["comparison_cue"]
            trial_type = standard_cue + "-->" + comparison_cue

            standard_stim = generate_stim(self.standard_center_frequency)
            if self.standard_center_frequency == self.comparison_center_frequency:
                comparison_stim = copy.deepcopy(standard_stim)
            else:
                comparison_stim = generate_stim(self.comparison_center_frequency)

            standard_stim = apply_cue(standard_stim, standard_cue, standard_angle, self.standard_center_frequency, head_radius=self.head_radius)
            comparison_stim = apply_cue(comparison_stim, comparison_cue, comparison_angle, self.comparison_center_frequency, head_radius=self.head_radius)

            standard_value = standard_stim.itd() if standard_cue == "ITD" else standard_stim.ild()
            comparison_value = comparison_stim.itd() if comparison_cue == "ITD" else comparison_stim.ild()

            datetime_onset = datetime.datetime.now()

            # Randomise presentation of the two stimuli
            presentations = [standard_stim, comparison_stim]
            order = np.random.permutation(len(presentations))
            for i, idx in enumerate(order):
                current_stim = presentations[idx]
                current_stim.play()
                if i == 0:
                    time.sleep(self.ISI)

            datetime_offset = datetime.datetime.now()

            response = get_response(seq_idx, len(trial_seq))

            reaction_time = datetime.datetime.now() - datetime_offset

            expected_standard = np.where(order == 0)[0][0]
            is_comparison_to_the_right = comparison_angle > standard_angle

            # Signal detection parameters
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

            # Scoring based on response
            score = None
            if is_comparison_to_the_right == 1:
                score = int(is_correct)
            elif is_comparison_to_the_right == 0:
                score = 1 - int(is_correct)
            comparison_angle_abs = comparison_angle * -1 if standard_angle < 0 else comparison_angle

            # 'Fold' scores over the symmetrical axis
            if standard_angle < 0 and score == 0:
                score_abs = score + 1
            elif standard_angle < 0 and score == 1:
                score_abs = score - 1
            else:
                score_abs = score

            # Save data into the results table object
            if save:
                row = results_table.Row(
                    subject=self.subject,
                    head_radius=self.head_radius,
                    datetime_onset=datetime_onset,
                    stim_type=self.stim_type,
                    trial_type=trial_type,
                    trial_index=seq_idx,
                    standard_angle=standard_angle,
                    standard_value=standard_value,  # (ms or dB) subject specific based on HRTF
                    standard_cue=standard_cue,
                    standard_center_frequency=self.standard_center_frequency,
                    comparison_angle=comparison_angle,
                    comparison_value=comparison_value,  # (ms or dB) subject specific based on HRTF
                    comparison_cue=comparison_cue,
                    comparison_center_frequency=self.comparison_center_frequency,
                    standard_order=np.where(order == 0)[0][0],
                    comparison_order=np.where(order == 1)[0][0],
                    solution=solution,
                    inter_stimulus_interval=self.ISI,
                    response=response,
                    is_correct=is_correct,
                    score=score,
                    score_abs=score_abs,
                    standard_angle_abs=abs(standard_angle),
                    reaction_time=reaction_time,
                    comparison_angle_abs=comparison_angle_abs
                )
                results_table.write(row)
            time.sleep(.3)
        print("Done with sequence!")
