import pathlib
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import slab

DIR = pathlib.Path(os.getcwd())
DIR_recs = DIR / "recordings"
subjects = os.listdir(DIR_recs)
subjects = [subject]

df_binaural_measurements = pd.DataFrame()
# freqs = [centre_frequency]

for subject in subjects:
    for azi in os.listdir(DIR_recs / subject):
        for freq in os.listdir(DIR_recs / subject / azi):
            for i, filename in enumerate(os.listdir(DIR_recs / subject / azi / freq)):
                rec = slab.Binaural(DIR_recs / subject / azi / freq / filename)
                azimuth_float = float(azi[4:])
                freq_float = float(freq)
                low_cutoff = freq_float / (2 ** (1 / 6))
                high_cutoff = freq_float * (2 ** (1 / 6))
                rec = rec.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
                itd = rec.itd() / rec.samplerate
                ild = rec.ild()
                binaural_current = {
                    "filename": filename,
                    "subject": subject,
                    "azimuth": azimuth_float,
                    "freq": freq_float,
                    "itd": itd,
                    "ild": ild,
                    "rms": rec.level.mean(),
                    "rms_left": rec.left.level,
                    "rms_right": rec.right.level
                }
                df_binaural_measurements = df_binaural_measurements.append(binaural_current, ignore_index=True)
                print("Computed binaural for", subject, "at", azi, "degrees","for rec", i + 1, "at", freq, 'Hz')