import time
from datetime import datetime
import slab
import freefield
import pathlib
import os
import pandas as pd
from analyze_binaural import get_binaural_data_row, save_binaural_data, process_binaural


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
    # freefield.write(tag="playbuflen", value=sound.n_samples, processors=["RX81", "RX82"])
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


def record_sounds(subject=None, speaker_ids=[2, 8, 15, 23, 31, 38, 44], freqs=[600, 800, 1000, 1200], n_reps=3):
    if subject is None:
        print("You haven't specified a subject")
    else:
        df_binaural_data = pd.DataFrame()
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
                    sound.level = 92

                    rec = play_and_record_local(sound, speaker)
                    filepath = get_rec_file_path(subject, azimuth, centre_frequency)

                    rec.write(filepath, normalise=False)

                    row = get_binaural_data_row(filepath, subject, azimuth, centre_frequency)
                    df_binaural_data = df_binaural_data.append(row, ignore_index=True)
                    time.sleep(0.1)
            print(f"Finished recording at azimuth angle: {azimuth}")
        df_post_proc_binaural_data = process_binaural(df_binaural_data)
        save_binaural_data(df_post_proc_binaural_data)
        print("Done saving binaural data")