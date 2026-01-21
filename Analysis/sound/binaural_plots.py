import pickle
import numpy as np
import matplotlib
matplotlib.use("MacOSX")  # for PyCharm on macOS
import matplotlib.pyplot as plt
import slab

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,   # TrueType fonts in PDF
    "ps.fonttype": 42
})

# --- paths ---
ILS_PATH = "ils_jakab.pickle"

# --- load ILS (interaural level spectrum) ---
with open(ILS_PATH, "rb") as f:
    ils = pickle.load(f)

# ils is expected to be a dict with keys like:
# 'frequencies' (n_bands,), 'azimuths' (n_az,), 'level_diffs_left' (n_bands, n_az), 'level_diffs_right' (n_bands, n_az)
# azimuths = np.asarray(ils["azimuths"], dtype=float)
azimuths = np.linspace(-30, 30, 1000)

# Frequencies you want
freqs_of_interest = [500, 1300, 1800]

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

# ---------- ILD from your ILS dict (via slab helper) ----------
# slab.Binaural.azimuth_to_ild returns a tuple of (left_level_dB, right_level_dB)
# From that we compute ILD = right - left (dB). (Positive => source to the right)
def ild_from_ils(azis_deg: np.ndarray, freq_hz: float, ils_dict: dict) -> np.ndarray:
    ild_vals = []
    for a in azis_deg:
        left_db, right_db = azimuth_to_ild(a, frequency=freq_hz, ils=ils_dict)
        ild_vals.append(right_db - left_db)
    return np.asarray(ild_vals, dtype=float)

# ---------- ITD from slab analytic mapping ----------
# slab.Binaural.azimuth_to_itd returns ITD in seconds (negative means source to the left)
def itd_from_slab(azis_deg: np.ndarray, freq_hz: float) -> np.ndarray:
    return np.asarray([slab.Binaural.azimuth_to_itd(a, frequency=freq_hz) for a in azis_deg], dtype=float)


# ---------- compute curves ----------
ild_curves = {f: ild_from_ils(azimuths, f, ils) for f in freqs_of_interest}
itd_curves = {f: itd_from_slab(azimuths, f) for f in freqs_of_interest}

# ---------- colour control (THIS IS THE KEY PART) ----------
blue_cmap   = plt.get_cmap("Blues")
orange_cmap = plt.get_cmap("Oranges")

# choose perceptually balanced intensities (avoid very light & very dark)
blue_levels   = [0.45, 0.65, 0.85]
orange_levels = [0.45, 0.55, 0.65]

# ---------- plotting ----------
fig, (ax_ild, ax_itd) = plt.subplots(
    1, 2, figsize=(10, 4), sharex=True
)

ax_ild.axhline(0, linewidth=1, ls="--", color="lightgrey")
ax_ild.axvline(0, linewidth=1, ls="--", color="lightgrey")
ax_itd.axhline(0, linewidth=1, ls="--", color="lightgrey")
ax_itd.axvline(0, linewidth=1, ls="--", color="lightgrey")

# --- ILD ---
for f, lvl in zip(freqs_of_interest, blue_levels):
    ax_ild.plot(
        azimuths,
        ild_curves[f],
        color=blue_cmap(lvl),
        label=f"{f} Hz"
    )


ax_ild.set_xlabel("Azimuth (deg)")
ax_ild.set_ylabel("ILD (dB)  [right − left]")
ax_ild.set_title("ILD vs Azimuth")
ax_ild.legend(frameon=False)

# --- ITD ---
for f, lvl in zip(freqs_of_interest, orange_levels):
    ax_itd.plot(
        azimuths,
        itd_curves[f] * 1e6,
        color=orange_cmap(lvl),
        label=f"{f} Hz"
    )


ax_itd.set_xlabel("Azimuth (deg)")
ax_itd.set_ylabel("ITD (µs)")
ax_itd.set_title("ITD vs Azimuth")
ax_itd.legend(frameon=False)

plt.tight_layout()
fig.savefig("Analysis/sound/ILD and ITD vs azimuth.png", dpi=300, bbox_inches="tight", format="png")
plt.show()