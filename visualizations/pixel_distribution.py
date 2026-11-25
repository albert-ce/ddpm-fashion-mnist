from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torchvision import datasets

# (row, col)
PIXEL = (7, 7)
# None to skip
PIXEL_PAIR = ((7, 7), (7, 20))
ROOT = Path(__file__).resolve().parent.parent
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
TIMESTEPS = [0, 20, 40, 80, 140, 200, 300, 1000]
VALUE_RANGE = (-3.5, 3.5)
N_BINS = 60
N_BINS_2D = 40
SEED = 0
OUT_DIR = ROOT / "figures" / "distributions"

OUT_DIR.mkdir(parents=True, exist_ok=True)


dataset = datasets.FashionMNIST(root=ROOT / "data", train=True, download=True)
images = dataset.data.double() / 255 * 2 - 1  # [-1, 1]
mean_image = images.mean(0).numpy()

traj = torch.load(TRAJECTORIES_PATH)
order = np.argsort(traj["timesteps"].numpy())
frame_ts = traj["timesteps"].numpy()[order]
frame_ab = traj["alphas_cumprod"].double().numpy()[order]
x_rev = traj["x"][order]  # (n_frames, n_samples, 1, 28, 28)
n_samples = x_rev.shape[1]
print(f"Loaded {n_samples} trajectories with {len(frame_ts)} frames from {TRAJECTORIES_PATH}")

missing = set(TIMESTEPS) - set(frame_ts.tolist())
assert not missing, f"t = {sorted(missing)} were not saved; available every {traj['snapshot_every']} steps"
shown = sorted(TIMESTEPS)
shown_idx = [frame_ts.tolist().index(t) for t in shown]
shown_ab = frame_ab[shown_idx]

rng = np.random.default_rng(SEED)
data_idx = rng.choice(len(images), n_samples, replace=False)


def pixel_values(pixels):
    rows, cols = zip(*pixels)
    clean = images[:, rows, cols].numpy()
    gen = x_rev[:, :, 0, rows, cols].float().numpy()
    return clean, gen


def forward_samples(clean, ab):
    x0 = clean[data_idx]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * rng.standard_normal(x0.shape)


def forward_density(clean, ab, *axes):
    # exact forward density
    values, counts = np.unique(clean, axis=0, return_counts=True)
    weights = counts / counts.sum()
    var = 1 - ab
    factors = [
        np.exp(-((a[:, None] - np.sqrt(ab) * values[:, p]) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for p, a in enumerate(axes)
    ]
    if len(factors) == 1:
        return factors[0] @ weights
    return (factors[0] * weights) @ factors[1].T  # (len(axis0), len(axis1))


def mark_pixels(ax, pixels, colors):
    ax.imshow(mean_image, cmap="gray", vmin=-1, vmax=1)
    for (r, c), color in zip(pixels, colors):
        ax.scatter([c], [r], s=60, facecolors="none", edgecolors=color, linewidths=2)
    ax.axis("off")


def row_labels(axes):
    axes[0, 0].text(-0.15, 0.5, "Forward\ndata + noise\n→", transform=axes[0, 0].transAxes, ha="right", va="center", fontsize=11)
    axes[1, 0].text(1.0, 0.5, "Reverse\ngenerated\n←", transform=axes[1, 0].transAxes, ha="right", va="center", fontsize=11)


row, col = PIXEL
clean_1d, gen_1d = pixel_values([PIXEL])
clean_1d, gen_1d = clean_1d[:, 0], gen_1d[..., 0]
print(f"Pixel {PIXEL}: {np.mean(clean_1d == -1):.1%} of training images are background (-1); generated x_0 outside [-1, 1]: {np.mean(np.abs(gen_1d[0]) > 1):.1%}")


bins = np.linspace(*VALUE_RANGE, N_BINS + 1)
grid = np.linspace(*VALUE_RANGE, 400)
n_cols = len(shown)
fig, axes = plt.subplots(2, n_cols + 1, figsize=(2.5 * (n_cols + 1), 5.5), squeeze=False)
mark_pixels(axes[0, 0], [PIXEL], ["red"])
axes[0, 0].set_title(f"pixel {PIXEL}\n(mean image)", fontsize=10)
axes[1, 0].axis("off")
row_labels(axes)

for i, (t, ab, f) in enumerate(zip(shown, shown_ab, shown_idx), start=1):
    top, bottom = axes[0, i], axes[1, i]
    top.hist(forward_samples(clean_1d, ab), bins=bins, density=True, color="tab:blue", alpha=0.6)
    bottom.hist(gen_1d[f], bins=bins, density=True, color="tab:orange", alpha=0.6)
    if ab < 1:
        for ax in (top, bottom):
            ax.plot(grid, forward_density(clean_1d[:, None], ab, grid), color="k", lw=1.2, label="exact forward density")
    top.set_title(f"t = {t}\n√ᾱ = {np.sqrt(ab):.2f}", fontsize=10)
    bottom.set_xlabel("pixel value")
    for ax in (top, bottom):
        ax.set(xlim=VALUE_RANGE, yticks=[])
        ax.axvline(-1, color="grey", lw=0.6, ls=":")
        ax.axvline(1, color="grey", lw=0.6, ls=":")
axes[1, -1].legend(fontsize=7, loc="upper right")
fig.suptitle(f"Marginal distribution of pixel {PIXEL} ({n_samples} values per panel; dotted = [-1, 1])")
fig.tight_layout()
out = OUT_DIR / f"pixel_{row}_{col}_histograms.png"
fig.savefig(out, dpi=150, transparent=True)
print(f"Saved {out}")


if PIXEL_PAIR is not None:
    (r1, c1), (r2, c2) = PIXEL_PAIR
    clean_2d, gen_2d = pixel_values(PIXEL_PAIR)
    corr = np.corrcoef(clean_2d.T)[0, 1]
    print(f"Pixels {PIXEL_PAIR}: correlation of the clean values {corr:.2f}")

    edges = np.linspace(*VALUE_RANGE, N_BINS_2D + 1)
    axis = np.linspace(*VALUE_RANGE, 200)

    fig, axes = plt.subplots(2, n_cols + 1, figsize=(2.6 * (n_cols + 1), 5.8), squeeze=False)
    mark_pixels(axes[0, 0], PIXEL_PAIR, ["red", "cyan"])
    axes[0, 0].set_title(f"red {PIXEL_PAIR[0]} → x\ncyan {PIXEL_PAIR[1]} → y", fontsize=9)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.15, f"clean corr = {corr:.2f}", ha="center", fontsize=9, transform=axes[1, 0].transAxes)
    row_labels(axes)

    for i, (t, ab, f) in enumerate(zip(shown, shown_ab, shown_idx), start=1):
        top, bottom = axes[0, i], axes[1, i]
        for ax, v, cm in [(top, forward_samples(clean_2d, ab), "Blues"), (bottom, gen_2d[f], "Oranges")]:
            ax.hist2d(v[:, 0], v[:, 1], bins=edges, cmap=cm, norm=LogNorm(), cmin=1)
        if ab < 1:
            D = forward_density(clean_2d, ab, axis, axis)
            levels = D.max() * np.array([1e-3, 1e-2, 1e-1])
            for ax in (top, bottom):
                ax.contour(axis, axis, D.T, levels=levels, colors="k", linewidths=0.6)
        top.set_title(f"t = {t}\n√ᾱ = {np.sqrt(ab):.2f}", fontsize=10)
        for ax in (top, bottom):
            ax.set(xlim=VALUE_RANGE, ylim=VALUE_RANGE, aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            ax.tick_params(labelsize=7)
            ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="grey", lw=0.6, ls=":")
    fig.suptitle(
        f"Joint distribution of pixels {PIXEL_PAIR[0]} (x) and {PIXEL_PAIR[1]} (y)  "
        f"({n_samples} points per panel; black = exact forward density contours; dotted = [-1, 1]²)"
    )
    fig.tight_layout()
    out = OUT_DIR / f"pixels_{r1}_{c1}_{r2}_{c2}_joint.png"
    fig.savefig(out, dpi=150, transparent=True)
    print(f"Saved {out}")
