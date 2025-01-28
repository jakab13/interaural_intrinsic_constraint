import time
from datetime import datetime
import slab
import freefield
import pathlib
import os
import pandas as pd


SAMPLE_RATE = 48828

slab.Signal.set_default_samplerate(SAMPLE_RATE)
DIR = pathlib.Path(os.getcwd())
recording_samplerate = 97656

proc_list = [['RP2', 'RP2',  DIR/'rcx'/'bi_rec_buf.rcx'],
                         ['RX81', 'RX8', DIR/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'rcx'/'play_buf.rcx']]

def initialize():
    freefield.initialize('dome', zbus=True, device=proc_list)


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


def get_rec_file_path(subject, azimuth, centre_frequency):
    folder = "recordings"
    filename = '_'.join(filter(None, (subject, "azi", str(azimuth), "freq", str(centre_frequency))))
    filepath = pathlib.Path(
        folder / pathlib.Path(subject) / str("azi_" + str(azimuth)) / str(centre_frequency) / pathlib.Path(filename +
                                                                                                           datetime.now().strftime(
                                                                                                               "_%Y-%m-%d-%H-%M-%S") + '.wav'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def get_binaural_data_row(filepath, subject, azimuth, centre_frequency):
    rec = slab.Binaural(filepath)
    low_cutoff = centre_frequency / (2 ** (1 / 6))
    high_cutoff = centre_frequency * (2 ** (1 / 6))
    rec_filt = rec.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
    itd = rec_filt.itd() / rec.samplerate
    ild = rec_filt.ild()
    rms = rec_filt.level.mean(),
    rms_left = rec_filt.left.level,
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


def record_sounds(subject=None, speaker_ids=[2, 8, 15, 23, 31, 38, 44], freqs=[400, 600, 800, 1000, 1200], n_reps=5):
    if subject is None:
        print("You haven't specified a subject")
    else:
        df_binaural_measurements = pd.DataFrame()
        for speaker_id in speaker_ids:
            speaker = freefield.pick_speakers(speaker_id)[0]
            azimuth = speaker.azimuth
            for centre_frequency in freqs:
                low_cutoff = centre_frequency / (2 ** (1 / 6))
                high_cutoff = centre_frequency * (2 ** (1 / 6))
                for _ in range(n_reps):
                    noise_band_limited = slab.Sound.whitenoise(duration=.5, level=90).filter(frequency=(low_cutoff, high_cutoff),
                                                                                             kind="bp").ramp()
                    sound = noise_band_limited
                    sound.level = 90

                    rec = play_and_record_local(sound, speaker)
                    filepath = get_rec_file_path(subject, azimuth, centre_frequency)

                    rec.write(filepath, normalise=False)

                    row = get_binaural_data_row(filepath, subject, azimuth, centre_frequency)
                    df_binaural_measurements = df_binaural_measurements.append(row, ignore_index=True)
                    time.sleep(0.1)
        data_folder = "binaural_data"
        data_filepath = pathlib.Path(data_folder / pathlib.Path(subject + datetime.now().strftime("_%Y-%m-%d-%H-%M-%S") + '.csv'))
        data_filepath.parent.mkdir(parents=True, exist_ok=True)
        df_binaural_measurements.to_csv(data_filepath)