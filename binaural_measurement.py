import time
from datetime import datetime
import slab
import freefield
import pathlib
import os

subject = "jakab"
azimuth = 15

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

noise_broadband = slab.Sound.whitenoise(duration=.5, level=90).ramp()
noise_band_limited = slab.Sound.whitenoise(duration=.5, level=90).filter(frequency=(low_cutoff, high_cutoff), kind="bp").ramp()
tone = slab.Sound.tone(duration=.5, level=90, frequency=centre_frequency).ramp()
silence = slab.Sound.silence(duration=.5)

sound = noise_broadband
sound.level = 90

stims = [noise_broadband, noise_band_limited, tone]

for stim in stims:
    for _ in range(10):
        rec = play_and_record_local(stim)
        itd = rec.itd()
        print("itd:", itd)
        folder = "recordings"
        filename = '_'.join(filter(None, (subject, "azi", str(azimuth), "stim_type")))
        filepath = pathlib.Path(folder / pathlib.Path(subject) / pathlib.Path(filename +
                                                                              datetime.now().strftime(
                                                                                  "_%Y-%m-%d-%H-%M-%S") + '.txt'))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        rec.write(filepath)
        time.sleep(0.1)


