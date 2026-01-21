# PyCharm-friendly interactive plot using Matplotlib Sliders (no Jupyter/ipywidgets required).
# - Move the sliders to change k1, k2, sigma1, sigma2 and the x-range.
# - Toggle between "Optimal fusion" and "Linear sum" models.
# - Each cue is drawn as y_i = k_i * x with a ±sigma_i band.
# - The fused line has slope and band given by the selected model.
#
# Notes:
# - Uses only matplotlib (no external widget packages).
# - Avoids specifying colors explicitly; uses defaults.
# - Close the figure to end the script when running from PyCharm.
#
# Model details:
#   Cue i: y_i = k_i * x + e_i,   e_i ~ N(0, sigma_i^2)
#   Fisher info: I_i = k_i^2 / sigma_i^2
#   Optimal fusion (measurement-space slope):
#       k12_opt = (k1^3/sigma1^2 + k2^3/sigma2^2) / (k1^2/sigma1^2 + k2^2/sigma2^2)
#       sd12_opt_x = 1/sqrt(I1 + I2)            # in x-units
#       sd12_opt_y = sd12_opt_x * k12_opt       # in y-units (for shading around fused line)
#   Linear sum:
#       y_12 = (k1 + k2) * x + (e1 + e2)
#       k12_sum = k1 + k2
#       sd12_sum_y = sqrt(sigma1^2 + sigma2^2)  # per interval, in y-units
#
#   We plot ±(sd in y-units) as the band around each line.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ----- initial parameters -----
k1_0, k2_0 = 0.9, 0.1
s1_0, s2_0 = 0.2, 0.3
xmin_0, xmax_0 = 0, 5.0
npts_0 = 401
model_0 = "Optimal fusion"  # or "Linear sum"

# ----- helper computations -----
def fused_params_optimal(k1, k2, s1, s2):
    I1 = (k1**2)/(s1**2) if s1 > 0 else np.inf
    I2 = (k2**2)/(s2**2) if s2 > 0 else np.inf
    Itot = I1 + I2
    if not np.isfinite(Itot) or Itot == 0:
        k12_y = np.nan
        sd12_y = np.nan
    else:
        k12_y = (k1**3/s1**2 + k2**3/s2**2) / (k1**2/s1**2 + k2**2/s2**2)
        sd12_x = 1.0 / np.sqrt(Itot)
        sd12_y = abs(k12_y) * sd12_x
    return k12_y, sd12_y, I1, I2, Itot

def fused_params_sum(k1, k2, s1, s2):
    k12_y = k1 + k2
    sd12_y = np.sqrt(s1**2 + s2**2)
    return k12_y, sd12_y

# ----- initial data -----
x = np.linspace(xmin_0, xmax_0, npts_0)

def compute_lines(k1, k2, s1, s2, x, model):
    y1 = k1 * x
    y2 = k2 * x

    if model == "Optimal fusion":
        k12, sd12, I1, I2, Itot = fused_params_optimal(k1, k2, s1, s2)
        info_text = f"I1={I1:.4g}, I2={I2:.4g}, I_total={Itot:.4g}"
    else:
        k12, sd12 = fused_params_sum(k1, k2, s1, s2)
        info_text = f"Linear-sum: sd12_y={sd12:.4g}"

    y12 = k12 * x if np.isfinite(k12) else np.full_like(x, np.nan)
    return y1, y2, y12, k12, sd12, info_text

# ----- figure and axes -----
plt.close('all')
fig = plt.figure(figsize=(9, 6))
gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.5)

ax_plot = fig.add_subplot(gs[:4, 0])

# Slider axes
ax_k1 = fig.add_subplot(gs[4, 0])
ax_k2 = ax_k1.inset_axes([0.0, -0.75, 1.0, 1.0])
ax_s1 = fig.add_subplot(gs[5, 0])
ax_s2 = ax_s1.inset_axes([0.0, -0.75, 1.0, 1.0])

# Radio buttons for model
rax = ax_plot.inset_axes([0.76, 0.60, 0.20, 0.20])  # position inside main plot
radio = RadioButtons(rax, ("Optimal fusion", "Linear sum"), active=0 if model_0=="Optimal fusion" else 1)

# Create sliders
sl_k1 = Slider(ax=ax_k1, label="k1", valmin=0.0, valmax=2.0, valinit=k1_0, valstep=0.01)
sl_k2 = Slider(ax=ax_k2, label="k2", valmin=0.0, valmax=2.0, valinit=k2_0, valstep=0.01)
sl_s1 = Slider(ax=ax_s1, label="σ1", valmin=0.001, valmax=2.0, valinit=s1_0, valstep=0.001)
sl_s2 = Slider(ax=ax_s2, label="σ2", valmin=0.001, valmax=2.0, valinit=s2_0, valstep=0.001)

# Initial lines
y1, y2, y12, k12, sd12, info_text = compute_lines(k1_0, k2_0, s1_0, s2_0, x, model_0)
ln1, = ax_plot.plot(x, y1, color="tab:blue", label=f"Cue 1: y=k1·x  (k1={k1_0:.3g}, σ1={s1_0:.3g})")
band1 = ax_plot.fill_between(x, y1 - s1_0, y1 + s1_0, color="tab:blue", alpha=0.15)

ln2, = ax_plot.plot(x, y2, color="tab:orange", label=f"Cue 2: y=k2·x  (k2={k2_0:.3g}, σ2={s2_0:.3g})")
band2 = ax_plot.fill_between(x, y2 - s2_0, y2 + s2_0, color="tab:orange", alpha=0.15)

ln12, = ax_plot.plot(x, y12, color="tab:green", linestyle="--", label=f"Fused: y=k12·x  (k12={k12:.3g}, sd_y={sd12:.3g})")
band12 = ax_plot.fill_between(x, y12 - sd12, y12 + sd12, color="tab:green", alpha=0.15)

# Annotations/labels
txt = ax_plot.text(0.02, 0.98, info_text, transform=ax_plot.transAxes, va="top", ha="left")
ax_plot.set_title("Two cues and their fused estimate")
ax_plot.set_xlabel("Physical stimulus x")
ax_plot.set_ylabel("Measurement / percept")
ax_plot.grid(True, alpha=0.25)
ax_plot.legend(loc="best", frameon=False)

def update_plot(val=None):
    k1 = sl_k1.val
    k2 = sl_k2.val
    s1 = sl_s1.val
    s2 = sl_s2.val
    model = radio.value_selected

    # recompute
    y1, y2, y12, k12, sd12, info_text = compute_lines(k1, k2, s1, s2, x, model)

    # update main lines
    ln1.set_ydata(y1)
    ln1.set_label(f"Cue 1: y=k1·x  (k1={k1:.3g}, σ1={s1:.3g})")
    ln2.set_ydata(y2)
    ln2.set_label(f"Cue 2: y=k2·x  (k2={k2:.3g}, σ2={s2:.3g})")
    ln12.set_ydata(y12)
    ln12.set_label(f"Fused: y=k12·x  (k12={k12:.3g}, sd_y={sd12:.3g})")

    # update bands: need to remove and redraw
    global band1, band2, band12
    for b in [band1, band2, band12]:
        try:
            b.remove()
        except Exception:
            pass

    band1 = ax_plot.fill_between(x, y1 - s1, y1 + s1, color="tab:blue", alpha=0.15)
    band2 = ax_plot.fill_between(x, y2 - s2, y2 + s2, color="tab:orange", alpha=0.15)
    band12 = ax_plot.fill_between(x, y12 - sd12, y12 + sd12, color="tab:green", alpha=0.15)

    # update info text
    txt.set_text(info_text)

    # refresh legend (labels might change)
    ax_plot.legend(loc="best", frameon=False)
    fig.canvas.draw_idle()

# Connect sliders and radio buttons
sl_k1.on_changed(update_plot)
sl_k2.on_changed(update_plot)
sl_s1.on_changed(update_plot)
sl_s2.on_changed(update_plot)
radio.on_clicked(update_plot)

plt.show()
