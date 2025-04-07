import matplotlib
matplotlib.use('MacOSX')
import copy
import slab
import numpy as np
import datetime
import time
import data_handler
from sound_handler import apply_cue

subject = "jakab_matched"

# standard_angle_conditions = [-4, -3, -2, -1, 0, 0, 1, 2, 3, 4]
standard_angle_conditions = [2]
# comparison_angle_conditions = np.asarray([-35, -25, -15, -5, 5, 15, 25, 35])
comparison_angle_conditions = np.asarray([-25, -15, -5, 5, 15, 25, 35])
standard_cue = "ITD"
comparison_cue = "ILD"
n_reps = 10