# visual_blobs_localisation_2AFC.py

from psychopy import visual, core, event, gui
import numpy as np
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. Experiment setup
# -----------------------------

exp_info = {
    "Participant": "",
    "Session": "001",
}

dlg = gui.DlgFromDict(exp_info, title="Visual Blobs Localisation 2AFC")
if not dlg.OK:
    core.quit()

exp_name = "visual_blobs_localisation_2AFC"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_base = f"{exp_name}_{exp_info['Participant']}_{timestamp}"

# Create data folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

csv_filename = os.path.join("data", file_base + ".csv")

# Window
win = visual.Window(
    size=[1280, 720],
    fullscr=False,
    color="black",
    units="pix"
)

trial_counter = visual.TextStim(
    win,
    text="",
    pos=(-550, 350),     # top-left corner for 1280x720 window in pix units
    height=30,
    color="white",
    anchorHoriz="left",
    anchorVert="top"
)

# Fixation cross
fixation = visual.TextStim(win, text="+", height=0.7, color="white")

# Instruction text
instructions = visual.TextStim(
    win,
    text=(
        "Press any key to start."
    ),
    height=0.7,
    wrapWidth=200,
    color="white"
)

instructions.draw()
win.flip()
event.waitKeys()

# -----------------------------
# 2. Stimulus definitions
# -----------------------------
# Blob parameters (same base size, different blur, pre-generated PNGs)
blob_files = {
    "sharp": "stimuli/blob_sharp.png",
    "blurry": "stimuli/blob_blurry.png"
}

# Common size on screen (in pixels)
BLOB_DRAW_SIZE = 600  # tweak as needed

def make_blob(blur_type, x_pos):
    img_path = blob_files[blur_type]
    stim = visual.ImageStim(
        win,
        image=img_path,
        size=BLOB_DRAW_SIZE,   # displayed size (width=height)
        pos=(x_pos, 0)         # center vertically
    )
    return stim

# -----------------------------
# 3. Trial design (constant stimuli)
# -----------------------------

# Δx values (in pixels) for comparison blob relative to reference
delta_positions = np.array([-8, -6, -4, -2, 2, 4, 6, 8])

# Number of repetitions per Δx per blur pairing
reps_per_condition = 20

# All blur pairings: (reference_blur, comparison_blur)
blur_types = ["sharp", "blurry"]
blur_pairings = [
    ("sharp", "sharp"),
    ("sharp", "blurry"),
    ("blurry", "sharp"),
    ("blurry", "blurry")
]

trials = []

for ref_blur, comp_blur in blur_pairings:
    for dx in delta_positions:
        for _ in range(reps_per_condition):
            trials.append({
                "ref_blur": ref_blur,
                "comp_blur": comp_blur,
                "delta_x": dx
            })

# Shuffle trials
np.random.shuffle(trials)

# Timing (s)
fix_duration = 0.5
isi = 0.3
stim_duration = 0.15

# -----------------------------
# 4. Run experiment
# -----------------------------

# Open CSV file
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header (extended to include order and actual positions)
    writer.writerow([
        "participant",
        "session",
        "ref_blur",
        "comp_blur",
        "delta_x",
        "order",       # "ref_first" or "comp_first"
        "first_type",  # "reference" or "comparison"
        "second_type", # "reference" or "comparison"
        "first_x",
        "second_x",
        "response",
        "correct",
        "rt",
    ])

    clock = core.Clock()
    all_trials = []  # also keep in memory for analysis
    # trial_counter.draw()
    for all_trial_index, t in enumerate(trials):
        ref_blur = t["ref_blur"]
        comp_blur = t["comp_blur"]
        delta_x = t["delta_x"]

        # Ideal positions if reference is at 0 and comparison at delta_x
        ref_x = np.random.choice([-4, 4])
        comp_x = ref_x + delta_x

        # Randomise which comes first: reference or comparison
        order = np.random.choice(["ref_first", "comp_first"])

        if order == "ref_first":
            first_type = "reference"
            second_type = "comparison"
            first_blur = ref_blur
            second_blur = comp_blur
            first_x = ref_x
            second_x = comp_x
        else:  # "comp_first"
            first_type = "comparison"
            second_type = "reference"
            first_blur = comp_blur
            second_blur = ref_blur
            first_x = comp_x
            second_x = ref_x

        # Create stimuli for this trial
        first_blob = make_blob(first_blur, first_x)
        second_blob = make_blob(second_blur, second_x)

        current_trial = all_trial_index + 1
        total_trials = len(trials)
        trial_counter.text = f"{current_trial} / {total_trials}"

        # Draw trial counter + fixation
        # trial_counter.draw()
        print(f"{current_trial} / {total_trials}")
        # 1) Fixation
        fixation.draw()
        win.flip()
        core.wait(fix_duration)

        # 2) FIRST stimulus
        first_blob.draw()
        win.flip()
        core.wait(stim_duration)

        # 3) ISI
        win.flip()
        core.wait(isi)

        # 4) SECOND stimulus
        second_blob.draw()
        win.flip()
        core.wait(stim_duration)

        # 5) Response screen
        question = visual.TextStim(
            win,
            text="",
            height=0.7,
            wrapWidth=500,
            color="white"
        )
        question.draw()
        win.flip()

        event.clearEvents()
        clock.reset()
        keys = event.waitKeys(
            keyList=["left", "right", "escape"],
            timeStamped=clock
        )

        key, rt = keys[0]

        if key == "escape":
            win.close()
            core.quit()

        response = key

        # Determine correct answer based on actual positions shown
        # If second_x > first_x, second blob is to the right of first.
        if second_x > first_x:
            correct_answer = "right"
        elif second_x < first_x:
            correct_answer = "left"
        else:
            correct_answer = None  # ambiguous (shouldn't happen unless delta_x == 0)

        if correct_answer is None:
            correct = ""
        else:
            correct = int(response == correct_answer)

        # Write to CSV
        writer.writerow([
            exp_info["Participant"],
            exp_info["Session"],
            ref_blur,
            comp_blur,
            delta_x,
            order,
            first_type,
            second_type,
            first_x,
            second_x,
            response,
            correct,
            rt
        ])

        # Store in memory
        all_trials.append({
            "ref_blur": ref_blur,
            "comp_blur": comp_blur,
            "delta_x": delta_x,
            "order": order,
            "first_type": first_type,
            "second_type": second_type,
            "first_x": first_x,
            "second_x": second_x,
            "response": response,
            "correct": correct,
            "rt": rt
        })

        # Optional short ITI
        win.flip()
        core.wait(0.2)

# End of experiment
thanks = visual.TextStim(win, text="Thank you!\n\nPress any key to see the results.", height=0.7, color="white")
thanks.draw()
win.flip()
event.waitKeys()

win.close()
core.quit()  # Comment this out if you want analysis below in same run in some environments

# -----------------------------
# 5. Offline analysis & plotting
# -----------------------------
# NOTE: In some environments, core.quit() will exit Python completely.
# If that happens, move the analysis to a separate script that reads the CSV.
# You can also comment out core.quit() above while debugging.

# If running analysis in the same script, comment out the core.quit() line above.
