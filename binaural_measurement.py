import slab
import freefield
import pathlib
import os

slab.Signal.set_default_samplerate(97656.25)
DIR = pathlib.Path(os.getcwd())
recording_samplerate = 195312.5

proc_list = [['RP2', 'RP2',  DIR/'rcx'/'bi_rec_buf.rcx'],
                         ['RX81', 'RX8', DIR/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'rcx'/'play_buf.rcx']]

freefield.initialize('dome', zbus=True, device=proc_list)


centre_frequency = 500

noise = slab.Sound.whitenoise(duration=1.)
tone = slab.Sound.tone(duration=1., frequency=centre_frequency, level=90)
speaker = 23

sound = tone

freefield.write(tag="playbuflen", value=sound.n_samples, processors=["RX81", "RX82"])
n_delay = freefield.get_recording_delay(play_from="RX8", rec_from="RP2")
n_delay += 50  # make the delay a bit larger to avoid missing the sound's onset
rec_n_samples = int(sound.duration * recording_samplerate)
freefield.write(tag="playbuflen", value=rec_n_samples + n_delay, processors="RP2")
freefield.set_signal_and_speaker(sound, speaker, equalize=False)
freefield.play()
freefield.wait_to_finish_playing()
rec_l = freefield.read(tag='datal', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
rec_r = freefield.read(tag='datar', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
rec = slab.Binaural([rec_l, rec_r], samplerate=recording_samplerate)

