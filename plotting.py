import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('MacOSX')


# Compute posterior parameters
def get_combined_sigma(sigma_prior, sigma_likelihood):
    sigma_combined = np.sqrt((sigma_prior**2 * sigma_likelihood**2) / (sigma_prior**2 + sigma_likelihood**2))
    return sigma_combined


def get_combined_location(mu_prior, sigma_prior, mu_likelihood, sigma_likelihood):
    mu_posterior = (mu_prior * sigma_likelihood**2 + mu_likelihood * sigma_prior**2) / (sigma_prior**2 + sigma_likelihood**2)
    return mu_posterior


def get_PSE(x, sigma_prior, sigma_likelihood):
    PSE = (sigma_prior**2 / sigma_likelihood**2) * x
    return PSE


def get_sigma(x, sigma_0, sigma_diff=0.):
    sigma = sigma_0 + sigma_diff * abs(x)
    return sigma


def get_cue_color(cue):
    return "tab:orange" if cue == "T" else "tab:blue"


# ITD cue manipulation
# Initial parameters
x_T = 10
x_L = 10
sigma_T_0 = 1.5
sigma_L_0 = 1.
sigma_T_diff = 0.03
sigma_L_diff = 0.01
standard_cue = "L"
comparison_cue = "T"

# Calculate standard deviations as a function of position
sigma_T = get_sigma(x_T, sigma_T_0, sigma_T_diff)
sigma_L = get_sigma(x_L, sigma_L_0, sigma_L_diff)

# Combined distribution parameters
x_combined = get_combined_location(x_T, sigma_T, x_L, sigma_L)
sigma_combined = get_combined_sigma(sigma_T, sigma_L)

# Define distributions in a given range
x_range = np.linspace(-2, 45, 1000)
dist_T = norm.pdf(x_range, x_T, sigma_T)
dist_L = norm.pdf(x_range, x_L, sigma_L)
dist_combined = norm.pdf(x_range, x_combined, sigma_combined)

# Plot distributions
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 7))
plt.subplots_adjust(bottom=0.15)

row_standard = ax[0]
ax_standard_dist = row_standard[0]
ax_standard_pred = row_standard[1]

row_combined = ax[1]
ax_combined_dist = row_combined[0]
ax_combined_pred = row_combined[1]

row_comparison = ax[2]
ax_comparison_dist = row_comparison[0]
ax_comparison_pred = row_comparison[1]


def plot_PDF(ax, x, sigma, cue, is_combined=False):
    dist = norm.pdf(x_range, x, sigma)
    c = get_cue_color(cue)
    ls = "-" if is_combined else "--"
    ax.plot(x_range, dist, label=f"{cue} ~N({x:.1f}, {sigma:.1f}²)", c=c, ls=ls)
    if is_combined:
        ax.plot(x, dist.max(), "o", markersize=4, c=c)


def plot_cue_PDFs(ax, cue_changed="T"):
    cue_unchanged = "L" if cue_changed == "T" else "T"
    x_changed = x_T if cue_changed == "T" else x_L
    x_unchanged = 0
    sigma_changed_0 = sigma_T_0 if cue_changed == "T" else sigma_L_0
    sigma_unchanged_0 = sigma_L_0 if cue_changed == "T" else sigma_T_0
    sigma_changed_diff = sigma_T_diff if cue_changed == "T" else sigma_L_diff
    sigma_unchanged_diff = sigma_L_diff if cue_changed == "T" else sigma_T_diff
    sigma_changed = get_sigma(x_changed, sigma_changed_0, sigma_changed_diff)
    sigma_unchanged = get_sigma(x_unchanged, sigma_unchanged_0, sigma_unchanged_diff)
    ax.vlines(0, ymin=0, ymax=1, color="grey", alpha=.5)
    plot_PDF(ax, x_changed, sigma_changed, cue_changed)
    plot_PDF(ax, x_unchanged, sigma_unchanged, cue_unchanged)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Presented location (x)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, color="lightgrey", alpha=.5)


def plot_combined_PDF(ax, cue_changed="T"):
    x_changed = x_T if cue_changed == "T" else x_L
    x_unchanged = 0
    sigma_changed_0 = sigma_T_0 if cue_changed == "T" else sigma_L_0
    sigma_unchanged_0 = sigma_L_0 if cue_changed == "T" else sigma_T_0
    sigma_changed_diff = sigma_T_diff if cue_changed == "T" else sigma_L_diff
    sigma_unchanged_diff = sigma_L_diff if cue_changed == "T" else sigma_T_diff
    sigma_changed = get_sigma(x_changed, sigma_changed_0, sigma_changed_diff)
    sigma_unchanged = get_sigma(x_unchanged, sigma_unchanged_0, sigma_unchanged_diff)
    x_combined = get_combined_location(x_changed, sigma_changed, x_unchanged, sigma_unchanged)
    sigma_combined = get_combined_sigma(sigma_changed, sigma_unchanged)
    ax.vlines(0, ymin=0, ymax=1, color="grey", alpha=.5)
    # plot_PDF(ax, x_combined, sigma_combined, cue_changed, is_combined=True)
    c = get_cue_color(cue_changed)
    pdf_1 = norm.pdf(x_range, x_changed, sigma_changed)
    pdf_2 = norm.pdf(x_range, x_unchanged, sigma_unchanged)
    product_pdf = pdf_1 * pdf_2
    product_pdf /= np.trapezoid(product_pdf, x_range)  # Normalize area to 1
    ax.plot(x_range, product_pdf, label=f"{cue_changed} ~N({x_combined:.1f}, {sigma_combined:.1f}²)", c=c)
    ax.plot(x_combined, product_pdf.max(), "o")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Presented location (x)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, color="lightgrey", alpha=.5)


def plot_prediction(ax, cue_changed="T"):
    x_changed = x_T if cue_changed == "T" else x_L
    x_unchanged = 0
    sigma_changed_0 = sigma_T_0 if cue_changed == "T" else sigma_L_0
    sigma_unchanged_0 = sigma_L_0 if cue_changed == "T" else sigma_T_0
    sigma_changed_diff = sigma_T_diff if cue_changed == "T" else sigma_L_diff
    sigma_unchanged_diff = sigma_L_diff if cue_changed == "T" else sigma_T_diff
    sigma_changed = get_sigma(x_range, sigma_changed_0, sigma_changed_diff)
    sigma_unchanged = get_sigma(x_unchanged, sigma_unchanged_0, sigma_unchanged_diff)
    x_combined_predicted = get_combined_location(x_range, sigma_changed, x_unchanged, sigma_unchanged)
    ax.plot(x_range, x_combined_predicted, color=get_cue_color(cue_changed), ls="--")
    ax.plot(x_changed, x_combined, "o", markersize=8, color=get_cue_color(cue_changed))
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 45)
    ax.set_xlabel("Presented location (x)")
    ax.set_ylabel("Perceived location (x̂)")
    ax.legend()
    ax.grid(True, color="lightgrey", alpha=.5)


plot_cue_PDFs(ax_standard_dist, cue_changed=standard_cue)
plot_combined_PDF(ax_combined_dist, cue_changed=standard_cue)
plot_prediction(ax_standard_pred, cue_changed=standard_cue)
plot_prediction(ax_combined_pred, cue_changed=standard_cue)

plot_cue_PDFs(ax_comparison_dist, cue_changed=comparison_cue)
plot_combined_PDF(ax_combined_dist, cue_changed=comparison_cue)
plot_prediction(ax_comparison_pred, cue_changed=comparison_cue)
plot_prediction(ax_combined_pred, cue_changed=comparison_cue)


fig_x_0, _ = ax_comparison_dist.transData.transform((0, 0))  # Convert x=0 to display coordinates
fig_x_1, _ = ax_comparison_dist.transData.transform((45, 0))  # Convert x=0 to display coordinates
slider_x_0, _ = fig.transFigure.inverted().transform((fig_x_0, 0))  # Convert to figure coords
slider_x_1, _ = fig.transFigure.inverted().transform((fig_x_1, 0))  # Convert to figure coords

ax_standard_slider = plt.axes([slider_x_0, 0.9, slider_x_1 - slider_x_0, 0.03])  # Position of slider
standard_slider = Slider(ax_standard_slider, f"Standard ({standard_cue})", 0, 45, valinit=x_T, color=get_cue_color(standard_cue))
ax_comparison_slider = plt.axes([slider_x_0, 0.05, slider_x_1 - slider_x_0, 0.03])  # Position of slider
comparison_slider = Slider(ax_comparison_slider, f"Comparison ({comparison_cue})", 0, 45, valinit=x_L, color=get_cue_color(comparison_cue))


def update(val, ax_dist, ax_combined, ax_pred, cue_changed="T", stim="standard"):
    x_changed = val
    sigma_changed_0 = sigma_T_0 if cue_changed == "T" else sigma_L_0
    sigma_unchanged_0 = sigma_L_0 if cue_changed == "T" else sigma_T_0
    sigma_changed_diff = sigma_T_diff if cue_changed == "T" else sigma_L_diff
    sigma_changed = get_sigma(x_changed, sigma_changed_0, sigma_changed_diff)
    x_combined = get_combined_location(x_changed, sigma_changed, 0, sigma_unchanged_0)
    sigma_combined = get_combined_sigma(sigma_changed, sigma_unchanged_0)

    # Update distributions
    dist_changed = norm.pdf(x_range, x_changed, sigma_changed)
    dist_combined = norm.pdf(x_range, x_combined, sigma_combined)

    # Update distributions
    line_cue_changed = ax_dist.get_lines()[0]
    line_cue_changed.set_ydata(dist_changed)
    line_cue_changed.set_label(f"{cue_changed} ~N({x_changed:.1f}, {sigma_changed:.1f}²)")

    # Update combined distribution
    line_cue_combined_standard = ax_combined.get_lines()[0]
    dot_x_combined_standard = ax_combined.get_lines()[1]
    line_cue_combined_comparison = ax_combined.get_lines()[2]
    dot_x_combined_comparison = ax_combined.get_lines()[3]

    line_cue_combined = line_cue_combined_standard if stim == "standard" else line_cue_combined_comparison
    dot_x_combined = dot_x_combined_standard if stim == "standard" else dot_x_combined_comparison
    line_cue_combined.set_ydata(dist_combined)
    dot_x_combined.set_data([x_combined], [dist_combined.max()])

    overlap = np.minimum(line_cue_combined_standard.get_ydata(), line_cue_combined_comparison.get_ydata())
    if ax_combined.collections:
        ax_combined.collections[0].remove()
    ovarlap_proportion = overlap.sum() / dist_combined.sum()
    ax_combined.fill_between(x_range, overlap, color="tab:red", label=f"Overlap: {ovarlap_proportion:.2f}")


    line_cue_combined.set_label(f"Combined ~N({x_combined:.1f}, {sigma_combined:.1f}²)")

    ax_dist.legend()
    ax_combined.legend()

    # Update perceived prediction
    dot_x_combined = ax_pred.get_lines()[1]
    dot_x_combined.set_data([x_changed], [x_combined])

    fig.canvas.draw_idle()


# Connect slider to update function
standard_slider.on_changed(lambda val: update(val,
                                              ax_standard_dist,
                                              ax_combined_dist,
                                              ax_standard_pred,
                                              cue_changed=standard_cue,
                                              stim="standard")
                           )
comparison_slider.on_changed(lambda val: update(val,
                                                ax_comparison_dist,
                                                ax_combined_dist,
                                                ax_comparison_pred,
                                                cue_changed=comparison_cue,
                                                stim="comparison")
                             )

# Show interactive plot
plt.show()



