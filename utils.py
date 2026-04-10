import numpy as np
import pickle

ILS_PATH = "ils_jakab_new.pickle"

# --- load ILS (interaural level spectrum) ---
with open(ILS_PATH, "rb") as f:
    ILS = pickle.load(f)

def get_response(trial_count=None, max_count=None):
    while True:
        if trial_count is not None and max_count is not None:
            response = input(f"Which way did the sound move? <----- 1 | 2 -----> ({trial_count + 1}/{max_count})")
        else:
            response = input(f"Which way did the sound move? <----- 1 | 2 ----->")
        if response == "49" or response == "1":
            response = "left"
            break
        elif response == "51" or response == "2":
            response = "right"
            break
        else:
            continue
    return response


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


def ild_from_ils(azis_deg: np.ndarray, freq_hz: float, ils_dict: dict) -> np.ndarray:
    ild_vals = []
    for a in azis_deg:
        left_db, right_db = azimuth_to_ild(a, frequency=freq_hz, ils=ils_dict)
        ild_vals.append(right_db - left_db)
        # ild_curr = slab.Binaural.azimuth_to_ild(a, frequency=freq_hz, ils=ils)
        # ild_vals.append(ild_curr)
    return np.asarray(ild_vals, dtype=float)


def azimuth_from_ild(
    target_ild_db,
    freq_hz,
    ils_dict=ILS,
    azimuth_grid_deg=None,
):
    """
    Find the azimuth angle (in degrees) at a given frequency that corresponds
    to a target ILD value.

    Parameters
    ----------
    target_ild_db : float
        Desired ILD value in dB (right - left).
    freq_hz : float
        Frequency at which to evaluate the ILD->azimuth mapping.
    ils_dict : dict
        ILS/HRTF dictionary used by azimuth_to_ild.
    azimuth_grid_deg : np.ndarray | None
        Azimuth grid to sample. If None, uses -90..90 in 0.1 deg steps.

    Returns
    -------
    float
        Estimated azimuth in degrees.

    Raises
    ------
    ValueError
        If the target ILD lies outside the sampled ILD range.
    """
    if azimuth_grid_deg is None:
        azimuth_grid_deg = np.linspace(-50, 50, 1801)  # 0.1 deg steps

    ild_vals = ild_from_ils(
        azis_deg=azimuth_grid_deg,
        freq_hz=freq_hz,
        ils_dict=ils_dict
    )

    # Make sure interpolation sees increasing x-values
    sort_idx = np.argsort(ild_vals)
    ild_sorted = ild_vals[sort_idx]
    az_sorted = azimuth_grid_deg[sort_idx]

    ild_min = ild_sorted.min()
    ild_max = ild_sorted.max()

    if not (ild_min <= target_ild_db <= ild_max):
        raise ValueError(
            f"Target ILD {target_ild_db:.2f} dB is outside the sampled range "
            f"[{ild_min:.2f}, {ild_max:.2f}] dB at {freq_hz} Hz."
        )

    azimuth_est = np.interp(target_ild_db, ild_sorted, az_sorted)
    return float(azimuth_est)
