from experiment import Experiment
from experiment_analysis import *

subject = "jakab_mg"

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
exp.mixing_gain = 0.75
exp.run_sequence()

# ===========================================================================
exp.mixing_gain = 1
exp.run_sequence()


# ===========================================================================
# Plotting ==================================================================

df = get_df(subject, exp.standard_center_frequency, exp.comparison_center_frequency)
sns.lmplot(df, x="comparison_angle_abs", y="score_abs", hue="mixing_gain", logistic=True)