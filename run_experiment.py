from experiment import Experiment
from experiment_analysis import *

subject = "jakab_test"

exp = Experiment(subject)

exp.standard_center_frequency = 800
exp.comparison_center_frequency = 800
exp.head_radius = 8.5  # in cm
reference_angle = 10
PSE_estimate = 5

# Familiarisation (without saving data)
exp.n_reps = 2
exp.run_sequence(save=False)

# Number of repetitions for one sequence
exp.n_reps = 4  # should be an even number

# ITD-->ITD =====================================
exp.standard_cue = "ITD"
exp.comparison_cue = "ITD"
exp.standard_angle = reference_angle
exp.PSE_angle = reference_angle
exp.run_sequence()

# Plot psychometric functions
plot_pfs(subject)

# ITD-->ILD =====================================
exp.standard_cue = "ITD"
exp.comparison_cue = "ILD"
exp.standard_angle = reference_angle
exp.PSE_angle = PSE_estimate
exp.run_sequence()

# Plot psychometric functions
plot_pfs(subject)

# Get PSE from above measurement
PSE_estimate = get_PSE(subject, "ITD", "ILD", exp.standard_angle, exp.standard_center_frequency)

# ILD-->ILD =====================================
exp.standard_cue = "ILD"
exp.comparison_cue = "ILD"
exp.standard_angle = PSE_estimate
exp.PSE_angle = PSE_estimate
exp.run_sequence()

plot_pfs(subject)

# ILD-->ITD =====================================
exp.standard_cue = "ILD"
exp.comparison_cue = "ITD"
exp.standard_angle = PSE_estimate
exp.PSE_angle = reference_angle
exp.run_sequence()

plot_pfs(subject)
