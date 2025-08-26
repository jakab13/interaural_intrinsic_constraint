import slab
import pickle
import numpy as np
import copy

ils = pickle.load(open('ils_jakab.pickle', 'rb'))


def make_interaural_level_spectrum():
    hrtf = slab.HRTF.kemar()  # load KEMAR by default
    # get the filters for the frontal horizontal arc
    idx = np.where((hrtf.sources.vertical_polar[:, 1] == 0) & (
        (hrtf.sources.vertical_polar[:, 0] <= 90) | (hrtf.sources.vertical_polar[:, 0] >= 270)))[0]
    # at this point, we could just get the transfer function of each filter with hrtf.data[idx[i]].tf(),
    # but it may be better to get the spectral left/right differences with ERB-spaced frequency resolution:
    azi = hrtf.sources.vertical_polar[idx, 0]
    # 270<azi<360 -> azi-360 to get negative angles on the left
    azi[azi >= 270] = azi[azi >= 270]-360
    sort = np.argsort(azi)
    fbank = slab.Filter.cos_filterbank(samplerate=hrtf.samplerate, pass_bands=True)
    freqs = fbank.filter_bank_center_freqs()
    noise = slab.Sound.pinknoise(duration=5., samplerate=hrtf.samplerate)
    noise_0 = slab.Binaural(hrtf.data[idx[0]].apply(noise))
    noise_0_bank = fbank.apply(noise_0.left)
    ils = dict()
    ils['samplerate'] = hrtf.samplerate
    ils['frequencies'] = freqs
    ils['azimuths'] = azi[sort]
    ils['level_diffs_right'] = np.zeros((len(freqs), len(idx)))
    ils['level_diffs_left'] = np.zeros((len(freqs), len(idx)))
    for n, i in enumerate(idx[sort]):  # put the level differences in order of increasing angle
        noise_filt = slab.Binaural(hrtf.data[i].apply(noise))
        noise_bank_left = fbank.apply(noise_filt.left)
        noise_bank_right = fbank.apply(noise_filt.right)
        ils['level_diffs_right'][:, n] = noise_0_bank.level - noise_bank_right.level
        ils['level_diffs_left'][:, n] = noise_0_bank.level - noise_bank_left.level
    return ils
#
#
# ils = make_interaural_level_spectrum()


def azimuth_to_ild(azimuth, frequency=2000, ils=None):
    level_diffs_left = ils['level_diffs_left']
    level_diffs_right = ils['level_diffs_right']
    # levels = [np.interp(azimuth, ils['azimuths'], level_diffs[i, :]) for i in range(level_diffs.shape[0])]
    levels_right = [np.interp(azimuth, ils['azimuths'], level_diffs_right[i, :]) for i in
                    range(level_diffs_right.shape[0])]
    levels_left = [np.interp(azimuth, ils['azimuths'], level_diffs_left[i, :]) for i in
                    range(level_diffs_left.shape[0])]
    ild_right = np.interp(frequency, ils['frequencies'], levels_right) * -1
    ild_left = np.interp(frequency, ils['frequencies'], levels_left) * -1
    return [ild_right, ild_left]  # interpolate level difference at frequency


def apply_cue(stim, cue, angle, filter_frequency, head_radius=8):
    sound = copy.deepcopy(stim)
    if cue == "ITD" or cue == "BOTH":
        itd_val = slab.Binaural.azimuth_to_itd(angle, frequency=filter_frequency, head_radius=head_radius)
        sound = sound.itd(itd_val)
    if cue == "ILD" or cue == "BOTH":
        # ild_val = slab.Binaural.azimuth_to_ild(angle, frequency=filter_frequency, ils=ils)
        # sound = sound.ild(ild_val).externalize()
        ild_vals = azimuth_to_ild(angle, frequency=filter_frequency, ils=ils)
        sound.level += ild_vals
    return sound


def generate_stim(center_frequency, mixing_gain=0.5, level=80):
    low_cutoff = center_frequency / (2 ** (1 / 6))
    high_cutoff = center_frequency * (2 ** (1 / 6))
    noise_1 = slab.Sound.whitenoise(duration=0.3, samplerate=44100)
    noise_2 = slab.Sound.whitenoise(duration=0.3, samplerate=44100)
    stim_l = mixing_gain * noise_1 + (1 - mixing_gain) * noise_2
    stim_r = (1 - mixing_gain) * noise_1 + mixing_gain * noise_2
    stim_l = stim_l.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
    stim_r = stim_r.filter(frequency=(low_cutoff, high_cutoff), kind="bp")
    stim = slab.Binaural.silence(duration=stim_l.n_samples, samplerate=44100)
    stim.left = stim_l.data[:, 0]
    stim.right = stim_r.data[:, 0]
    stim = stim.ramp(duration=0.01)
    stim.level = level
    level_aweight = stim.aweight().level
    level_diff = stim.level - level_aweight
    stim.level += level_diff
    stim.level -= 3
    return stim
