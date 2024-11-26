# PARTICIPANT CHARACTERISATION (HRTF-like measurements) #

# measurements in the freefield
# put cap on participant and calibrate the laser
# insert in-ear microphones in participant
# record sine-wave sounds on the two mic channels
# repeat multiple times (at least 3, more like 5)
# repeat for multiple loudnesses (?)
# repeat for different angles
# save recordings

#################### PRE-PROCESSING #######################

# load recordings per participant, per frequency, per loudness
# average across repetitions within conditions
# calculate ILD and ITD for each condition
# save values in table

# HRFT data structure
# subject
# head_radius
# frequency
# loudness
# ILD
# ITD


#################### EXPERIMENT #######################

# load HRTF-like file to get subject specific ILDs and ITDs

# define standard angles
# define comparison angles
# define cue types (itd, ild)
# define number of repetitions per condition (#n of comparison angles)
# define standard center frequency
# define comparison center frequency


# generate trial sequence
#   permutation of standard angles, comparison angles, cue types repeated n times
#   shuffle order


# iterate through trial sequence
#   create standard sound
#   create comparison sound (copy standard)
#   apply cue to standard (inputs: subject, cue type, angle, frequency)
#   apply cue to comparison
#   define stimulus order (standard/comparison, first/second)
#   present first stimulus
#   wait for inter stimulus interval
#   present second stimulus
#   wait for response
#   collect response
#   check if correct
#   advance trial counter
#   save response


# Data structure

# subject
# datetime_onset
# stim_type (sine, noise, click?)
# trial_type (ITD-->ILD, ILD-->ITD, etc)
# trial_index
# standard_angle
# standard_value (ms or dB)
# standard_cue (ITD or ILD)
# standard_center_frequency
# comparison_angle
# comparison_value (ms or dB)
# comparison_cue (ITD or ILD)
# comparison_center_frequency
# standard_order
# comparison_order
# inter_stimulus_interval
# response
# reaction_time
# is_correct
# score


#################### POST-PROCESSING #######################


# define psychometric function shape (sigmoid)
# define inverse of the psychometric function (x = ln...)

# load dataset

# ..........