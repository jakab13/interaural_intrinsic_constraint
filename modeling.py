import numpy as np
import pandas as pd
import scipy.integrate as integrate
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('MacOSX')


# Compute posterior parameters
def get_combined_sigma(sigma_prior, sigma_likelihood):
    sigma_combined = np.sqrt((sigma_prior**2 * sigma_likelihood**2) / (sigma_prior**2 + sigma_likelihood**2))
    return sigma_combined


def get_combined_location(mu_prior, sigma_prior, mu_likelihood, sigma_likelihood):
    mu_posterior = (mu_prior * sigma_likelihood**2 + mu_likelihood * sigma_prior**2) / (sigma_prior**2 + sigma_likelihood**2)
    return mu_posterior


def get_combined_dist(mu_prior, sigma_prior, mu_likelihood, sigma_likelihood):
    pdf_prior = norm.pdf(x_range, mu_prior, sigma_prior)
    pdf_likeliehood = norm.pdf(x_range, mu_likelihood, sigma_likelihood)
    combined_pdf = pdf_prior * pdf_likeliehood
    combined_pdf /= np.trapezoid(combined_pdf, x_range)
    return combined_pdf


def get_sigma(x, sigma_0, sigma_diff=0.):
    sigma = sigma_0 + sigma_diff * abs(x)
    return sigma


def find_nearest_idx(val, arr):
    return np.abs(arr - val).argmin() - 1


def get_d_prime(mu1, sigma1,  mu2, sigma2):
    d_prime = np.abs(mu1 - mu2) / np.sqrt((sigma1 ** 2 + sigma2 ** 2) / 2)
    return d_prime


def get_d_prime_adjusted(mu1, sigma1,  mu2, sigma2):
    d_prime = get_d_prime(mu1, sigma1, mu2, sigma2)
    adjustment = np.sqrt(1 + (sigma1 ** 2 / sigma2 ** 2))
    d_prime_adjusted = d_prime / adjustment
    return d_prime_adjusted


def get_d_prime_bayes(mu1, sigma1, mu2, sigma2):
    # Calculate Bayes discriminability index d_B
    d_prime_bayes = np.sqrt((2 * (mu1 - mu2)**2) / (sigma1**2 + sigma2**2))
    return d_prime_bayes


def get_jnd_bayes(sigma1, sigma2, criterion_db=1.35):
    # Calculate JND using the Bayesian discriminability index
    jnd = np.sqrt((criterion_db**2 * (sigma1**2 + sigma2**2)) / 2)
    return jnd


# Compute probability of correct discrimination (PCD)
def get_PCD(mu1, sigma1,  mu2, sigma2):
    d_prime = get_d_prime(mu1, sigma1,  mu2, sigma2)
    adjustment = np.sqrt(1 + (sigma1**2 / sigma2**2))
    PCD = norm.cdf(d_prime / adjustment)  # Probability of correct discrimination
    return PCD


def get_cue_color(cue):
    return "tab:orange" if cue == "T" else "tab:blue"


fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
df = pd.DataFrame()
x_T_loop = np.linspace(1, 16, 16)
sigma_T_0_loop = np.linspace(1, 1.5, 5)
sigma_T_diff_loop = np.linspace(0.0, 0.02, 5)

# for sigma_T_0 in sigma_T_0_loop:
#     for sigma_T_diff in sigma_T_diff_loop:
#         for x_T in x_T_loop:
for fig_idx, standard_cue, comparison_cue in zip([0, 1, 2, 3], ["T", "L", "L", "T"], ["T", "L", "T", "L"]):
    x_T = 16
    # standard_cue = "L"
    # comparison_cue = "T"

    x_range = np.linspace(0, 35, 5000)

    sigma_T_0 = 1.6
    sigma_L_0 = 0.9
    sigma_T_diff = 0.01
    sigma_L_diff = 0.00

    sigma_T_range = get_sigma(x_range, sigma_T_0, sigma_T_diff)
    sigma_L_range = get_sigma(x_range, sigma_L_0, sigma_L_diff)
    y_T_range = get_combined_location(x_range, sigma_T_range, 0, sigma_L_0)
    y_L_range = get_combined_location(x_range, sigma_L_range, 0, sigma_T_0)
    y_T = y_T_range[find_nearest_idx(x_T, x_range)]
    x_L = x_range[find_nearest_idx(y_T, y_L_range)]
    sigma_T = get_sigma(x_T, sigma_T_0, sigma_T_diff)
    sigma_L = get_sigma(x_L, sigma_L_0, sigma_L_diff)

    x_standard = x_T if standard_cue == "T" else x_L
    sigma_standard = sigma_T if standard_cue == "T" else sigma_L
    sigma_standard_0 = sigma_T_0 if standard_cue == "T" else sigma_L_0
    sigma_standard_diff = sigma_T_diff if standard_cue == "T" else sigma_L_diff
    sigma_comparison_0 = sigma_L_0 if comparison_cue == "L" else sigma_T_0
    sigma_comparison_diff = sigma_L_diff if comparison_cue == "L" else sigma_T_diff
    y_standard_range = y_T_range if standard_cue == "T" else y_L_range
    y_comparison_range = y_L_range if comparison_cue == "L" else y_T_range
    sigma_standard_range = sigma_T_range if standard_cue == "T" else sigma_L_range
    sigma_comparison_range = sigma_L_range if comparison_cue == "L" else sigma_T_range

    # Get the PSE of ILD at a given ITD
    y_standard = y_standard_range[find_nearest_idx(x_standard, x_range)]
    y_PSE = y_standard
    x_PSE = x_range[find_nearest_idx(y_PSE, y_standard_range)]

    sigma_PSE = get_sigma(x_PSE, sigma_standard_0, sigma_standard_diff)

    # Find JNDs
    y_JND_bayes_range = get_jnd_bayes(sigma_PSE, sigma_comparison_range)
    y_JND_bayes = y_JND_bayes_range[find_nearest_idx(x_PSE, x_range)]
    y_JND = y_PSE + y_JND_bayes
    x_JND = x_range[find_nearest_idx(y_JND, y_comparison_range)]
    sigma_JND = get_sigma(x_JND, sigma_comparison_0, sigma_comparison_diff)

    # plt.figure(2)
    # plt.plot(d_prime_range, label=f"{standard_cue}-->{comparison_cue}")
    # plt.hlines(1, xmin=0, xmax=len(x_range), color="lightgrey", alpha=.1)
    # plt.legend()

    # print(f"At {x_T} with Standard: {standard_cue} - Comparison: {comparison_cue} JND = {(x_JND - x_T)} ")

    row = {
        "ref_angle": x_T,
        "standard_cue": standard_cue,
        "comparison_cue": comparison_cue,
        "trial_type": standard_cue + "-->" + comparison_cue,
        "sigma_T_0": sigma_T_0,
        "sigma_L_0": sigma_L_0,
        "sigma_T_diff": sigma_T_diff,
        "sigma_L_diff": sigma_L_diff,
        "JND": x_JND - x_T
    }
    df_row = pd.DataFrame([row])
    df = pd.concat([df, df_row], ignore_index=True)

    pdf_standard = norm.pdf(x_range, y_PSE, sigma_PSE)
    pdf_comparison = norm.pdf(x_range, y_JND, sigma_JND)

    PCD = get_PCD(y_PSE, sigma_PSE, y_JND, sigma_JND)

    overlap = np.minimum(pdf_standard, pdf_comparison)
    ovarlap_proportion = overlap.sum() / pdf_comparison.sum()

    c_standard = get_cue_color(standard_cue)
    c_comparison = get_cue_color(comparison_cue)

    plot_spec = outer[fig_idx]
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                             subplot_spec=plot_spec,
                                             width_ratios=[1, 4],
                                             wspace=0,
                                             hspace=0.1
                                             )
    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1])

    # Plot standard distribution
    ax1.plot(pdf_standard, x_range, c=c_standard)
    ax1.hlines(y_standard, xmin=0, xmax=pdf_standard.max(), color=c_standard, alpha=.3)
    # Plot comparison distribution
    ax1.plot(pdf_comparison, x_range, c=c_comparison)
    ax1.hlines(y_JND, xmin=0, xmax=pdf_comparison.max(), color=c_comparison, alpha=.3)
    # Plot overlapping area
    ax1.fill_between(overlap, x_range, facecolor="tab:red", alpha=.3, label=f"{ovarlap_proportion:.2}")
    # Axis parameters
    ax1.set_ylabel("Perceived location of combined stimulus (x̂)")
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, x_range.max())
    ax1.legend()
    ax1.invert_xaxis()

    fig.add_subplot(ax1)

    # Axis of perceptual functions
    ax2.plot(x_range, x_range, c="lightgrey", alpha=.5, ls="--")

    # # T - Perceptual function
    ax2.plot(x_range, y_T_range, c=get_cue_color("T"), ls=":")
    ax2.fill_between(x_range, y_T_range + sigma_T_range, y_T_range - sigma_T_range, color=get_cue_color("T"), alpha=.05)

    # # L - Perceptual function
    ax2.plot(x_range, y_L_range, c=get_cue_color("L"), ls=":")
    ax2.fill_between(x_range, y_L_range + sigma_L_range, y_L_range - sigma_L_range, color=get_cue_color("L"), alpha=.05)

    ax2.hlines(y_standard, xmin=-5, xmax=x_standard, color=c_standard, alpha=.3)
    ax2.hlines(y_JND, xmin=-5, xmax=x_JND, color=c_comparison, alpha=.3)

    # Starting location
    if standard_cue != "T":
        ax2.plot(x_T, y_T, "o", c=get_cue_color("T"), markersize=10, fillstyle="none", alpha=.5)
        ax2.vlines(x_T, ymin=0, ymax=y_T, color=get_cue_color("T"), alpha=.3)
        ax2.hlines(y_PSE, xmin=x_PSE, xmax=x_T, ls="--", color="lightgrey")

    # PSE
    ax2.vlines(x_standard, ymin=0, ymax=y_standard, color=c_standard, alpha=.3)
    ax2.plot(x_PSE, y_PSE, "o", c=c_standard, label=f"Standard ({standard_cue})", markersize=10)
    # ax2.text(x_PSE, y_PSE + 0.5, "Standard", ha="center", va="bottom")
    ax2.vlines(x_PSE, ymin=0, ymax=y_PSE, color=c_standard, alpha=.3)

    # JND
    ax2.plot(x_JND, y_JND, "P", c=c_comparison, label=f"Comparison ({comparison_cue})", markersize=10)
    ax2.vlines(x_JND, ymin=0, ymax=y_JND, color=c_comparison, alpha=.3)
    ax2.hlines(0, xmin=x_T, xmax=x_JND, color="tab:red", lw=10, label=f"JND={(x_JND - x_T):.2f}")
    #
    # Parameters
    ax2.set_xlim(0, x_range.max())
    ax2.set_ylim(0, x_range.max())
    ax2.set_xlabel("Presented location of single cue (x)")
    ax2.legend()
    ax2.tick_params(left=False, labelleft=False)
    fig.add_subplot(ax2)
    # plt.tight_layout()

fig.show()
# plt.savefig("JNDs at different cue combinations", dpi=200)

# g = sns.FacetGrid(data=df, col="sigma_T_diff", row="sigma_T_0", hue="trial_type", hue_order=["L-->L", "T-->T", "L-->T", "T-->L"])
# g.map(sns.lineplot, "ref_angle", "JND")