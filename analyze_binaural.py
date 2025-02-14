import pathlib
import os
from datetime import datetime
import pandas as pd
import slab

DIR = pathlib.Path(os.getcwd())
DIR_recs = DIR / "recordings"

def get_binaural_data_row(filepath, subject, azimuth, centre_frequency):
    rec = slab.Binaural(filepath)
    low_cutoff = centre_frequency / (2 ** (1 / 6))
    high_cutoff = centre_frequency * (2 ** (1 / 6))
    rec_filt = rec.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
    max_lag = 0.001
    max_lag = slab.Sound.in_samples(max_lag, rec_filt.samplerate)
    itd = rec_filt.itd()
    itd = itd / rec.samplerate if itd != max_lag else None
    ild = rec_filt.ild()
    rms = rec_filt.level.mean()
    rms_left = rec_filt.left.level
    rms_right = rec_filt.right.level
    binaural_current = {
        "sound_filename": filepath.name,
        "subject": subject,
        "azimuth": azimuth,
        "freq": centre_frequency,
        "itd": itd,
        "ild": ild,
        "rms": rms,
        "rms_left": rms_left,
        "rms_right": rms_right
    }
    return binaural_current

def save_binaural_data(df):
    subject = df["subject"].unique()[0]
    data_folder = "binaural_data"
    data_filepath = pathlib.Path(pathlib.Path(data_folder) / pathlib.Path(subject) / pathlib.Path(subject + datetime.now().strftime("_%Y-%m-%d-%H-%M-%S") + '.csv'))
    data_filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_filepath, index=False)


def get_binaural_data(subject):
    data_folder = "binaural_data"
    data_filepath = os.listdir(pathlib.Path(data_folder) / pathlib.Path(subject))[0]
    df = pd.read_csv(pathlib.Path(data_folder) / pathlib.Path(subject) / data_filepath)
    return df

def process_binaural(df_binaural):
    def norm_rms(row, channel="left"):
        subject = row["subject"]
        frequency = row["freq"]
        rms_name = "rms_" + channel
        rms_base = df_binaural[(df_binaural.subject == subject) & (df_binaural.freq == frequency) & (df_binaural.azimuth == 0)][rms_name].mean()
        rms_val = row[rms_name] - rms_base
        return rms_val

    def norm_itd(row, channel="left"):
        subject = row["subject"]
        frequency = row["freq"]
        itd_base = df_binaural[(df_binaural.subject == subject) & (df_binaural.freq == frequency) & (df_binaural.azimuth == 0)]["itd"].mean()
        itd_val = row["itd"] - itd_base if row["itd"] is not None else None
        return itd_val

    df_binaural["rms_left_norm"] = df_binaural.apply(lambda row: norm_rms(row, channel="left"), axis=1)
    df_binaural["rms_right_norm"] = df_binaural.apply(lambda row: norm_rms(row, channel="right"), axis=1)
    df_binaural["itd_norm"] = df_binaural.apply(lambda row: norm_itd(row), axis=1)
    return df_binaural

def run_analysis(subject=None):
    if subject is None:
        subjects = os.listdir(DIR_recs)
    else:
        subjects = [subject]
    df_binaural_data = pd.DataFrame()
    for subject in subjects:
        for azi in os.listdir(DIR_recs / subject):
            for freq in os.listdir(DIR_recs / subject / azi):
                for i, filename in enumerate(os.listdir(DIR_recs / subject / azi / freq)):
                    azimuth_float = float(azi[4:])
                    freq_float = float(freq)
                    filepath = DIR_recs / subject / azi / freq / filename
                    row = get_binaural_data_row(filepath, subject, azimuth_float, freq_float)
                    df_binaural_data = df_binaural_data.append(row, ignore_index=True)
                    print("Computed binaural for", subject, "at", azi, "degrees","for rec", i + 1, "at", freq, 'Hz')

        df_post_proc_binaural_data = process_binaural(df_binaural_data)
        save_binaural_data(df_post_proc_binaural_data)