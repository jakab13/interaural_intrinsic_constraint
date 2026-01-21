import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# ============================================
# 0. Set CSV file to analyse
# ============================================
csv_filename = "data/visual_blobs_localisation_2AFC_jakab_20251128_233031.csv"   # <-- CHANGE THIS

# ============================================
# 1. Load data
# ============================================
trials = []

with open(csv_filename, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            delta_x = float(row["delta_x"])
        except ValueError:
            continue

        ref_blur = row["ref_blur"]
        comp_blur = row["comp_blur"]

        try:
            correct = int(row["correct"]) if delta_x > 0 else 1- int(row["correct"])
        except ValueError:
            # skip rows with empty / invalid correct field
            continue

        trials.append({
            "ref_blur": ref_blur,
            "comp_blur": comp_blur,
            "delta_x": delta_x,   # signed
            "correct": correct
        })

if not trials:
    raise RuntimeError("No valid trials loaded. Check csv_filename and column names.")

# ============================================
# 2. Psychometric function: cumulative Gaussian
# ============================================
# We model P_correct as a function of |Δx|:
#   P_correct(|x|) = Phi( (|x| - mu) / sigma )

def cum_gauss_abs(x, mu, sigma):
    x = np.abs(x)
    return norm.cdf((x - mu) / sigma)

def cum_gauss(x, mu, sigma):
    x = np.array(x)
    return norm.cdf((x - mu) / sigma)

def fit_psychometric_abs(x_vals, p_correct):
    """
    Fit P_correct(|x|) with a cumulative Gaussian.
    x_vals should be non-negative (magnitudes).
    Returns (mu, sigma).
    """
    x_vals = np.array(x_vals)
    p_correct = np.array(p_correct)

    # Initial guesses
    mu0 = np.median(x_vals)
    sigma0 = (max(x_vals) - min(x_vals)) / 4.0 if len(x_vals) > 1 else 1.0

    popt, pcov = curve_fit(
        lambda x, mu, sigma: cum_gauss(x, mu, sigma),
        x_vals,
        p_correct,
        p0=[mu0, sigma0],
        maxfev=10000
    )
    return popt  # mu, sigma

# ============================================
# 3. Group data by blur pairing
# ============================================
blur_pairings = sorted(set((t["ref_blur"], t["comp_blur"]) for t in trials))

results = []

plt.figure()
plt.title("Psychometric functions (% correct)\nAll blur pairings")
plt.xlabel("Δx (pixels) [comparison - reference]")
plt.ylabel("Proportion correct")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)

colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

for idx, (ref_blur, comp_blur) in enumerate(blur_pairings):
    color = colors[idx % len(colors)]

    subset = [t for t in trials if t["ref_blur"] == ref_blur and t["comp_blur"] == comp_blur]
    if not subset:
        continue

    # ---------- signed data for plotting ----------
    signed_deltas = sorted(set(t["delta_x"] for t in subset))

    x_signed = []
    p_correct_signed = []

    for dx in signed_deltas:
        ts = [t for t in subset if t["delta_x"] == dx]
        if not ts:
            continue
        correct_vals = [t["correct"] for t in ts]
        if not correct_vals:
            continue
        p_corr = np.mean(correct_vals)
        x_signed.append(dx)
        p_correct_signed.append(p_corr)

    x_signed = np.array(x_signed)
    p_correct_signed = np.array(p_correct_signed)

    plt.scatter(
        x_signed,
        p_correct_signed,
        color=color,
        alpha=0.7,
        label=f"Data ref={ref_blur}, comp={comp_blur}" if idx == 0 else None
    )

    # ---------- collapse to |Δx| for fitting ----------
    abs_deltas = sorted(set(abs(t["delta_x"]) for t in subset))

    x_abs = []
    p_correct_abs = []

    for dx_abs in abs_deltas:
        ts = [t for t in subset if abs(t["delta_x"]) == dx_abs]
        if not ts:
            continue
        correct_vals = [t["correct"] for t in ts]
        if not correct_vals:
            continue
        p_corr = np.mean(correct_vals)
        x_abs.append(dx_abs)
        p_correct_abs.append(p_corr)

    if len(x_abs) < 2:
        print(f"Not enough levels to fit for pairing {ref_blur}-{comp_blur}. Skipping fit.")
        continue

    x_abs = np.array(x_abs)
    p_correct_abs = np.array(p_correct_abs)

    # ---------- fit cumulative Gaussian in |Δx| ----------
    try:
        mu, sigma = fit_psychometric_abs(x_signed, p_correct_signed)
    except Exception as e:
        print(f"Fit failed for pairing {ref_blur}-{comp_blur}: {e}")
        continue

    # Thresholds:
    #   x50 = mu      (50% correct)
    #   x84 ≈ mu + sigma
    x50 = mu
    x84 = mu + sigma
    JND = x84 - x50   # = sigma

    results.append({
        "ref_blur": ref_blur,
        "comp_blur": comp_blur,
        "mu": mu,
        "sigma": sigma,
        "x50": x50,
        "x84": x84,
        "JND": JND
    })

    # ---------- plot symmetric fit in signed space ----------
    x_fit = np.linspace(min(x_signed) * 1.2, max(x_signed) * 1.2, 400)
    y_fit = cum_gauss(x_fit, mu, sigma)

    plt.plot(
        x_fit,
        y_fit,
        color=color,
        linestyle="-",
        label=f"Fit ref={ref_blur}, comp={comp_blur} (JND={JND:.2f})"
    )

# tidy legend
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), loc="lower right")

plt.show()

# ============================================
# 4. Print summary
# ============================================
print("\n===== Thresholds and JNDs per blur pairing =====")
for r in results:
    print(
        f"ref={r['ref_blur']}, comp={r['comp_blur']}  |  "
        f"x50 = {r['x50']:.3f},  x84 = {r['x84']:.3f},  "
        f"JND (x84 - x50) = {r['JND']:.3f}"
    )
