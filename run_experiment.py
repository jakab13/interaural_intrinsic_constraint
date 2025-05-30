from experiment import Experiment
from experiment_analysis import *

subject = "mister_mu"

exp = Experiment(subject)

exp.standard_center_frequency = 1800
exp.comparison_center_frequency = 1800
exp.head_radius = 7.0  # in cm
reference_angle = 5
# PSE_estimate = 2  # initial guess of PSE

# Familiarisation (without saving data)
# exp.n_reps = 1
# exp.run_sequence(save=False)

# Number of repetitions for one sequence
exp.n_reps = 4  # should be an even number

# ITD-->ITD =====================================
exp.standard_cue = "ITD"
exp.comparison_cue = "ITD"
exp.standard_angle = reference_angle
exp.PSE_angle = reference_angle
exp.run_sequence()

ask_to_continue()

# Plot psychometric functions
# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency)

# ITD-->ILD =====================================
# exp.standard_cue = "ITD"
# exp.comparison_cue = "ILD"
# exp.standard_angle = reference_angle
# exp.PSE_angle = PSE_estimate
# exp.run_sequence()

# Plot psychometric functions
# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency)

# Get PSE from above measurement
PSE_estimate = get_PSE(subject, "ITD", "ILD", reference_angle, exp.standard_center_frequency)

# ILD-->ILD =====================================
exp.standard_cue = "ILD"
exp.comparison_cue = "ILD"
exp.standard_angle = PSE_estimate
exp.PSE_angle = PSE_estimate
exp.run_sequence()

ask_to_continue()

# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency)

# ILD-->ITD =====================================
exp.standard_cue = "ILD"
exp.comparison_cue = "ITD"
exp.standard_angle = PSE_estimate
exp.PSE_angle = reference_angle
exp.run_sequence()

plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency)

# Plot with JND values
# plot_pfs(subject, exp.standard_center_frequency, exp.comparison_center_frequency, plot_parameter=False, plot_JND=True)

