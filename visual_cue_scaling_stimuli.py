# make_advanced_blobs.py
import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

os.makedirs("stimuli", exist_ok=True)

# Global image settings
N = 800          # image size (pixels, width = height)
RADIUS = 200      # base circle radius (pixels)
CENTER = (N / 2, N / 2)


def save_rgba_with_alpha(alpha_img, filename, base_intensity=255, noise_img=None):
    """
    Helper: compose an RGBA image from an alpha channel and optional noise in RGB.
    alpha_img: PIL 'L' image (0-255)
    noise_img: optional PIL 'L' or 'RGB' to use as intensity; otherwise use white.
    """
    if noise_img is None:
        rgb = Image.new("RGB", (N, N), (base_intensity, base_intensity, base_intensity))
    else:
        if noise_img.mode == "L":
            rgb = Image.merge("RGB", (noise_img, noise_img, noise_img))
        else:
            rgb = noise_img.convert("RGB")

    rgba = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    rgba.paste(rgb, (0, 0), mask=alpha_img)
    rgba.save(os.path.join("stimuli", filename))


# -------------------------------
# 1) Base circular mask (hard edge)
# -------------------------------
def make_hard_circle_alpha(radius):
    y, x = np.indices((N, N))
    cx, cy = CENTER
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = (r <= radius).astype(np.float32)
    alpha = (mask * 255).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


# -------------------------------
# 2) Gaussian-edge blobs (sharp / blurry)
# -------------------------------
def make_gaussian_blob(radius, blur_sigma, filename):
    alpha = make_hard_circle_alpha(radius)
    if blur_sigma > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    save_rgba_with_alpha(alpha, filename)


# -------------------------------
# 3) Noisy blobs (internal luminance noise)
# -------------------------------
def make_noisy_blob(radius, blur_sigma, noise_std, filename):
    # Start from hard-edged circle mask
    alpha = make_hard_circle_alpha(radius)
    if blur_sigma > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    # Create noise image (grayscale)
    # Start around mid-high intensity and add Gaussian noise
    base = 220
    noise = np.random.normal(loc=base, scale=noise_std, size=(N, N)).astype(np.float32)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise, mode="L")

    save_rgba_with_alpha(alpha, filename, noise_img=noise_img)


# -------------------------------
# 4) Edge-jittered blobs
# -------------------------------
def make_edge_jittered_blob(radius, jitter_amp, jitter_freq, blur_sigma, filename):
    """
    radius: base radius
    jitter_amp: amplitude of radial jitter (pixels)
    jitter_freq: number of "waves" around the circle
    blur_sigma: blur applied to alpha afterwards
    """
    cx, cy = CENTER

    # Generate a jittered contour in polar coordinates
    n_points = 720  # resolution around the circle
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # sinusoidal jitter + small random jitter
    sinusoid = jitter_amp * np.sin(jitter_freq * angles)
    random_jitter = np.random.normal(0, jitter_amp * 0.3, size=n_points)
    r = radius + sinusoid + random_jitter

    # Convert polar to Cartesian
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)

    # Create a hard polygon mask
    alpha = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(alpha)
    polygon_points = list(zip(xs, ys))
    draw.polygon(polygon_points, fill=255)

    # Blur edges a bit for smoother contour
    if blur_sigma > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    save_rgba_with_alpha(alpha, filename)


# ===============================
# Generate stimuli
# ===============================

# 1) Base Gaussian blobs (like your existing sharp / blurry)
print("Generating base Gaussian blobs...")
make_gaussian_blob(RADIUS, blur_sigma=8, filename="blob_sharp.png")
make_gaussian_blob(RADIUS, blur_sigma=60, filename="blob_blurry.png")

# 2) Noisy blobs (internal noise)
print("Generating noisy blobs...")
make_noisy_blob(RADIUS, blur_sigma=10, noise_std=200, filename="blob_noisy_sharp.png")
make_noisy_blob(RADIUS, blur_sigma=40, noise_std=200, filename="blob_noisy_blurry.png")

# 3) Edge-jittered blobs
print("Generating edge-jittered blobs...")
make_edge_jittered_blob(
    radius=RADIUS,
    jitter_amp=2,      # how much the radius wiggles
    jitter_freq=200,     # number of bumps around the circle
    blur_sigma=60,
    filename="blob_edgejitter_sharpish.png"
)
make_edge_jittered_blob(
    radius=RADIUS,
    jitter_amp=15,
    jitter_freq=12,
    blur_sigma=3.0,
    filename="blob_edgejitter_veryjittery.png"
)

print("Done. Check the 'stimuli' folder.")
