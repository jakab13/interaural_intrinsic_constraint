import time
from datetime import datetime
import seaborn as sns
import pandas as pd
import slab
import freefield
import pathlib
import os
import matplotlib.pyplot as plt

subject = "kemar_inner_mic"
azimuth = 0
freqs = [400, 600, 800, 1000]

SAMPLE_RATE = 48828

slab.Signal.set_default_samplerate(SAMPLE_RATE)
DIR = pathlib.Path(os.getcwd())
recording_samplerate = 97656

proc_list = [['RP2', 'RP2',  DIR/'rcx'/'bi_rec_buf.rcx'],
                         ['RX81', 'RX8', DIR/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'rcx'/'play_buf.rcx']]

freefield.initialize('dome', zbus=True, device=proc_list)


def list_dir_local(path):
    current_list = [f for f in os.listdir(path) if not f.startswith(".")]
    current_list = [f for f in current_list if "Icon" not in f]
    return current_list

def play_and_record_local(sound, speaker):
    freefield.write(tag="playbuflen", value=sound.n_samples, processors=["RX81", "RX82"])
    n_delay = freefield.get_recording_delay(play_from="RX8", rec_from="RP2")
    n_delay += 50  # make the delay a bit larger to avoid missing the sound's onset
    rec_n_samples = int(sound.duration * recording_samplerate)
    freefield.write(tag="playbuflen", value=rec_n_samples + n_delay, processors="RP2")
    freefield.set_signal_and_speaker(sound, speaker, equalize=False)
    freefield.play()
    freefield.wait_to_finish_playing(proc="RX81")
    rec_l = freefield.read(tag='datal', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
    rec_r = freefield.read(tag='datar', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
    rec = slab.Binaural([rec_l, rec_r], samplerate=recording_samplerate)
    return rec



speaker_ids = [2, 8, 15, 23, 31, 38, 44]

for speaker_id in speaker_ids:
    speaker = freefield.pick_speakers(speaker_id)[0]
    azimuth = speaker.azimuth
    for centre_frequency in freqs:
        low_cutoff = centre_frequency / (2 ** (1 / 6))
        high_cutoff = centre_frequency * (2 ** (1 / 6))
        for _ in range(10):
            noise_broadband = slab.Sound.whitenoise(duration=.5, level=90).ramp()
            noise_band_limited = slab.Sound.whitenoise(duration=.5, level=90).filter(frequency=(low_cutoff, high_cutoff),
                                                                                     kind="bp").ramp()
            sound = noise_band_limited
            sound.level = 90

            rec = play_and_record_local(sound, speaker)
            folder = "recordings"
            filename = '_'.join(filter(None, (subject, "azi", str(azimuth), "freq", str(centre_frequency))))
            filepath = pathlib.Path(folder / pathlib.Path(subject) / str("azi_" + str(azimuth)) / str(centre_frequency) / pathlib.Path(filename +
                                                                                  datetime.now().strftime(
                                                                                      "_%Y-%m-%d-%H-%M-%S") + '.wav'))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            rec.write(filepath, normalise=False)
            time.sleep(0.1)

DIR_recs = DIR / "recordings"
subjects = [f for f in os.listdir(DIR_recs) if not f.startswith(".")]
# subjects = [subject]

df_binaural_measurements = pd.DataFrame()
# freqs = [centre_frequency]

for subject in subjects:
    for azi in list_dir_local(DIR_recs / subject):
        for freq in list_dir_local(DIR_recs / subject / azi):
            for i, filename in enumerate(list_dir_local(DIR_recs / subject / azi / freq)):
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
                df_binaural_measurements = pd.concat([df_binaural_measurements, pd.DataFrame([binaural_current])], ignore_index=True)
                print("Computed binaural for", subject, "at", azi, "degrees","for rec", i + 1, "at", freq, 'Hz')


# df_binaural_measurements.to_csv("binaural_measurements.csv")

df_binaural_measurements = pd.read_csv("binaural_measurements.csv")

# df_binaural_measurements[df_binaural_measurements.azimuth == 0].groupby(["subject", "freq"])["ild", "itd"].mean()
# df_binaural_measurements.groupby(["subject", "freq", "azimuth"])["ild", "itd"].mean()

def norm_rms(row, channel="left"):
    subject = row.subject
    freq = row.freq
    rms_name = "rms_" + channel
    rms_base = df_binaural_measurements[(df_binaural_measurements.subject == subject) &
                                       (df_binaural_measurements.freq == freq) &
                                       (df_binaural_measurements.azimuth == 0)][rms_name].mean()
    rms_val = row[rms_name] - rms_base
    return rms_val

df_binaural_measurements["rms_left_norm"] = df_binaural_measurements.apply(lambda row: norm_rms(row, channel="left"), axis=1)
df_binaural_measurements["rms_right_norm"] = df_binaural_measurements.apply(lambda row: norm_rms(row, channel="right"), axis=1)

df_binaural_measurements.loc[df_binaural_measurements.azimuth == -17.5, "azimuth"] = -8.725
# df_binaural_measurements.loc[df_binaural_measurements.azimuth == 17.5, "azimuth"] = 8.725

df_binaural_measurements_high = df_binaural_measurements[df_binaural_measurements.freq > 700]
df_binaural_measurements_low = df_binaural_measurements[df_binaural_measurements.freq < 700]

g = sns.FacetGrid(data=df_binaural_measurements_high, row="freq", col="subject")
# g.map(sns.lineplot, "azimuth", "rms_left_norm", marker="o", color="g")
g.map(sns.lineplot, "azimuth", "rms_right_norm", marker="o", color="r")
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements, hue="freq", col="subject", palette="copper")
g.map(sns.lineplot, "azimuth", "ild", marker="o")
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements, hue="freq", col="subject", palette="copper")
g.map(sns.lineplot, "azimuth", "itd", marker="o")
g.add_legend()
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements[df_binaural_measurements.freq < 700], hue="subject")
g.map(sns.lineplot, "azimuth", "itd", marker="o")
g.add_legend()
plt.show()

df_sub = df_binaural_measurements[df_binaural_measurements.subject == subject]

fig, ax = plt.subplots(5, 2)
sns.lineplot(data=df_sub, x="azimuth", y="rms_left", hue="freq", ax=ax[0], marker="o")
sns.lineplot(data=df_sub, x="azimuth", y="rms_right", hue="freq", ax=ax[1], marker="o")
plt.show()

sns.lineplot(data=df_sub, x="azimuth", y="ild", hue="freq", marker="o")
plt.show()

