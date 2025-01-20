import time
from datetime import datetime
import seaborn as sns
import pandas as pd
import slab
import freefield
import pathlib
import os
import matplotlib.pyplot as plt

subject = "kemar_mic"
azimuth = 35
freqs = [400, 500, 600, 700, 800]

SAMPLE_RATE = 48828

slab.Signal.set_default_samplerate(SAMPLE_RATE)
DIR = pathlib.Path(os.getcwd())
recording_samplerate = 97656

proc_list = [['RP2', 'RP2',  DIR/'rcx'/'bi_rec_buf.rcx'],
                         ['RX81', 'RX8', DIR/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'rcx'/'play_buf.rcx']]

freefield.initialize('dome', zbus=True, device=proc_list)


def play_and_record_local(sound):
    speaker = 23
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


centre_frequency = 500
low_cutoff = centre_frequency / (2 ** (1 / 6))
high_cutoff = centre_frequency * (2 ** (1 / 6))

noise_band_limited = slab.Sound.whitenoise(duration=.5, level=90).filter(frequency=(low_cutoff, high_cutoff), kind="bp").ramp()
tone = slab.Sound.tone(duration=.5, level=90, frequency=centre_frequency).ramp()
silence = slab.Sound.silence(duration=.5)

for _ in range(10):
    noise_broadband = slab.Sound.whitenoise(duration=.5, level=90).ramp()
    sound = noise_broadband
    sound.level = 80

    rec = play_and_record_local(sound)
    folder = "recordings"
    filename = '_'.join(filter(None, (subject, "azi", str(azimuth))))
    filepath = pathlib.Path(folder / pathlib.Path(subject) / str("azi_" + str(azimuth)) / pathlib.Path(filename +
                                                                          datetime.now().strftime(
                                                                              "_%Y-%m-%d-%H-%M-%S") + '.wav'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    rec.write(filepath)
    time.sleep(0.1)

DIR_recs = DIR / "recordings"
subjects = os.listdir(DIR_recs)
subjects = [subject]

df_binaural_measurements = pd.DataFrame()

for subject in subjects:
    for azi in os.listdir(DIR_recs / subject):
        for i, filename in enumerate(os.listdir(DIR_recs / subject / azi)):
            rec = slab.Binaural(DIR_recs / subject / azi / filename)
            azimuth_int = int(azi[4:])
            for freq in freqs:
                low_cutoff = freq / (2 ** (1 / 6))
                high_cutoff = freq * (2 ** (1 / 6))
                rec = rec.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
                itd = rec.itd() / rec.samplerate
                ild = rec.ild()
                binaural_current = {
                    "filename": filename,
                    "subject": subject,
                    "azimuth": azimuth_int,
                    "freq": freq,
                    "itd": itd,
                    "ild": ild,
                    "rms": rec.level.mean(),
                    "rms_left": rec.left.level,
                    "rms_right": rec.right.level
                }
                df_binaural_measurements = df_binaural_measurements.append(binaural_current, ignore_index=True)
                print("Computed binaural for", subject, "at", azi, "degrees","for rec", i + 1, "at", freq, 'Hz')


df_binaural_measurements[df_binaural_measurements.azimuth == 0].groupby(["subject", "freq"])["ild", "itd"].mean()
df_binaural_measurements.groupby(["subject", "freq", "azimuth"])["ild", "itd"].mean()

g = sns.FacetGrid(data=df_binaural_measurements, row="freq", col="subject", sharey=False)
g.map(sns.lineplot, "azimuth", "rms_left", marker="o", color="g")
g.map(sns.lineplot, "azimuth", "rms_right", marker="o", color="r")
g.map(sns.lineplot, "azimuth", "ild", marker="o", color="b")
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements, row="freq", col="subject")
g.map(sns.lineplot, "azimuth", "ild", marker="o", color="b")
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements, hue="freq", col="subject")
g.map(sns.lineplot, "azimuth", "itd", marker="o")
plt.show()

df_sub = df_binaural_measurements[df_binaural_measurements.subject == subject]

fig, ax = plt.subplots(5, 2)
sns.lineplot(data=df_sub, x="azimuth", y="rms_left", hue="freq", ax=ax[0], marker="o")
sns.lineplot(data=df_sub, x="azimuth", y="rms_right", hue="freq", ax=ax[1], marker="o")
plt.show()

sns.lineplot(data=df_sub, x="azimuth", y="ild", hue="freq", marker="o")
plt.show()