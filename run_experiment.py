from experiment import Experiment
from experiment_analysis import *

subject = "jakab_test_ild_both"

exp = Experiment(subject)

exp.standard_center_frequency = 200
exp.comparison_center_frequency = 200
exp.head_radius = 8.0  # in cm
reference_angle = 10
PSE_estimate = 4  # initial guess of PSE

# Familiarisation (without saving data)
# exp.n_reps = 1
# exp.run_sequence(save=False)

# Number of repetitions for one sequence
exp.n_reps = 4  # should be an even number

# ITD-->ITD =====================================
exp.standard_cue = "ILD"
exp.comparison_cue = "ILD"
exp.standard_angle = reference_angle
exp.PSE_angle = reference_angle
exp.run_sequence()

ask_to_continue()

# Plot psychometric functions
# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency, weak_cue="ILD", strong_cue="BOTH")

# ITD-->BOTH =====================================
exp.standard_cue = "ILD"
exp.comparison_cue = "BOTH"
exp.standard_angle = reference_angle
exp.PSE_angle = PSE_estimate
exp.run_sequence()

# Plot psychometric functions
# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency, weak_cue="ILD", strong_cue="BOTH")

# Get PSE from above measurement
PSE_estimate = get_PSE(subject, "ILD", "BOTH", reference_angle, exp.standard_center_frequency)

# BOTH-->BOTH =====================================
exp.standard_cue = "BOTH"
exp.comparison_cue = "BOTH"
exp.standard_angle = PSE_estimate
exp.PSE_angle = PSE_estimate
exp.run_sequence()

ask_to_continue()

# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency, weak_cue="ILD", strong_cue="BOTH")

# BOTH-->ILD =====================================
exp.standard_cue = "BOTH"
exp.comparison_cue = "ILD"
exp.standard_angle = PSE_estimate
exp.PSE_angle = reference_angle
exp.run_sequence()

plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency, weak_cue="ILD", strong_cue="BOTH")

# Plot with JND values
# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency, weak_cue="ILD", strong_cue="BOTH",
#          plot_parameter=False, plot_JND=True)

