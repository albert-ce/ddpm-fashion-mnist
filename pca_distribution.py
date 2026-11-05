"""
PCA view of the forward and reverse diffusion processes.

1. Fit a PCA on every image of the original FashionMNIST train set (pixel space, [-1, 1]).
   This PCA is fixed and used for every projection below.
2. Top row: real images noised with the forward process at several t, from the clean data (t = 0)
   to pure noise (t = T). Read left to right.
3. Bottom row: generated x_t from `sample_trajectories.py` at the same t, so each column compares
   what x_t should look like (top) with what the model produces (bottom). Read right to left.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torchvision import datasets

N_COMPONENTS = 2  # PCA dimension (the figure shows the first two components)
TRAJECTORIES_PATH = Path("trajectories.pt")  # written by sample_trajectories.py
TIMESTEPS = None  # list of t to show (must be saved frames), or None to pick them automatically
N_COLUMNS = 8  # number of t when picked automatically
SEED = 0
OUT_DIR = Path("pca_figures")

assert N_COMPONENTS >= 2, "The figure needs at least two components"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- data + PCA
dataset = datasets.FashionMNIST(root="./data", train=True, download=True)
# Same preprocessing as training: ToTensor -> [0, 1], Normalize(0.5, 0.5) -> [-1, 1]
X = dataset.data.reshape(len(dataset), -1).double() / 255 * 2 - 1  # (60000, 784)

mean = X.mean(0)
cov = (X - mean).T @ (X - mean) / (len(X) - 1)
eigvals, eigvecs = torch.linalg.eigh(cov)
order = eigvals.argsort(descending=True)
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
W = eigvecs[:, :N_COMPONENTS]  # (784, k)
explained = (eigvals[:N_COMPONENTS] / eigvals.sum()).numpy()


def project(x):
    """(N, ...) images in [-1, 1] -> (N, k) PCA coordinates."""
    x = x.reshape(len(x), -1).double()
    return ((x - mean) @ W).numpy()


Z_data = project(X)
print("Explained variance ratio:", np.round(explained, 4), "total:", explained.sum().round(4))
print("Std of the data along each PC:", np.round(np.sqrt(eigvals[:N_COMPONENTS].numpy()), 3))
# Pure noise x ~ N(0, I) lands on N(-W^T mean, I_k) in PCA space
print("Centre of N(0, I) in PCA space:", np.round((-mean @ W).numpy(), 3), "(std 1 in every component)")


# ---------------------------------------------------------------- generated trajectories
traj = torch.load(TRAJECTORIES_PATH)
frame_ts = traj["timesteps"].tolist()
frame_ab = traj["alphas_cumprod"].double()
n_samples = traj["x"].shape[1]
print(f"Loaded {n_samples} trajectories with {len(frame_ts)} frames from {TRAJECTORIES_PATH}")

if TIMESTEPS is None:
    # Evenly spaced in sqrt(alpha_bar), the weight of x_0 in x_t: in PCA space the data cloud shrinks
    # by the same amount between consecutive columns. Evenly spaced t would waste most columns on
    # t > 500, where x_t is already almost pure noise.
    signal = frame_ab.sqrt()
    targets = torch.linspace(signal.max().item(), signal.min().item(), N_COLUMNS)
    shown = sorted({frame_ts[(signal - v).abs().argmin()] for v in targets})
else:
    missing = set(TIMESTEPS) - set(frame_ts)
    assert not missing, f"t = {sorted(missing)} were not saved; available every {traj['snapshot_every']} steps"
    shown = sorted(TIMESTEPS)
shown_ab = [frame_ab[frame_ts.index(t)].item() for t in shown]
print("Timesteps shown:", shown)

Z_rev = [project(traj["x"][frame_ts.index(t)]) for t in shown]

# Forward process on as many real images as generated samples, so both rows have the same density scale
rng = torch.Generator().manual_seed(SEED)
X0 = X[torch.randperm(len(X), generator=rng)[:n_samples]]
Z_fwd = []
for ab in shown_ab:
    eps = torch.randn(X0.shape, generator=rng, dtype=X0.dtype)
    Z_fwd.append(project(np.sqrt(ab) * X0 + np.sqrt(1 - ab) * eps))


# ---------------------------------------------------------------- figure
all_z = np.concatenate([Z_data, *Z_rev, *Z_fwd])[:, :2]
lo, hi = np.percentile(all_z, [0.2, 99.8], axis=0)
pad = 0.05 * (hi - lo)
extent = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
gridsize = 45


def density(ax, z, cmap):
    # Clean data in light grey underneath; same grid, so the hexagons line up
    ax.hexbin(Z_data[:, 0], Z_data[:, 1], gridsize=gridsize, extent=extent, cmap=ListedColormap(["#e0e0e0"]), mincnt=1)
    ax.hexbin(z[:, 0], z[:, 1], gridsize=gridsize, extent=extent, bins="log", cmap=cmap, mincnt=1)
    ax.set(xlim=extent[:2], ylim=extent[2:], xticks=[], yticks=[])


n_cols = len(shown)
fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 6), squeeze=False)
for col, (t, ab, zf, zr) in enumerate(zip(shown, shown_ab, Z_fwd, Z_rev)):
    density(axes[0, col], zf, "Blues")
    density(axes[1, col], zr, "Oranges")
    axes[0, col].set_title(f"t = {t}\n√ᾱ = {np.sqrt(ab):.2f}", fontsize=10)
for ax, label in [(axes[0, 0], "Forward\ndata + noise\n→"), (axes[1, 0], "Reverse\ngenerated\n←")]:
    ax.set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center", labelpad=10)
fig.suptitle(
    f"Forward vs reverse diffusion in the data PCA "
    f"(PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%}; grey = clean data, {n_samples} points per panel)"
)
fig.tight_layout()
out = OUT_DIR / "forward_reverse_pca.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
