from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torchvision import datasets

N_COMPONENTS = 2
ROOT = Path(__file__).resolve().parent.parent
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
TIMESTEPS = None
N_COLUMNS = 8
SEED = 0
OUT_DIR = ROOT / "figures" / "distributions"

assert N_COMPONENTS >= 2, "The figure needs at least two components"
OUT_DIR.mkdir(parents=True, exist_ok=True)


dataset = datasets.FashionMNIST(root=ROOT / "data", train=True, download=True)
X = dataset.data.reshape(len(dataset), -1).double() / 255 * 2 - 1  # (60000, 784)

mean = X.mean(0)
cov = (X - mean).T @ (X - mean) / (len(X) - 1)
eigvals, eigvecs = torch.linalg.eigh(cov)
order = eigvals.argsort(descending=True)
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
W = eigvecs[:, :N_COMPONENTS]  # (784, k)
explained = (eigvals[:N_COMPONENTS] / eigvals.sum()).numpy()


def project(x):
    x = x.reshape(len(x), -1).double()
    return ((x - mean) @ W).numpy()


Z_data = project(X)
print("Explained variance ratio:", np.round(explained, 4), "total:", explained.sum().round(4))
print("Std of the data along each PC:", np.round(np.sqrt(eigvals[:N_COMPONENTS].numpy()), 3))
print("Centre of N(0, I) in PCA space:", np.round((-mean @ W).numpy(), 3), "(std 1 in every component)")


traj = torch.load(TRAJECTORIES_PATH)
frame_ts = traj["timesteps"].tolist()
frame_ab = traj["alphas_cumprod"].double()
n_samples = traj["x"].shape[1]
print(f"Loaded {n_samples} trajectories with {len(frame_ts)} frames from {TRAJECTORIES_PATH}")

if TIMESTEPS is None:
    # evenly spaced in sqrt(alpha_bar)
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

rng = torch.Generator().manual_seed(SEED)
X0 = X[torch.randperm(len(X), generator=rng)[:n_samples]]
Z_fwd = []
for ab in shown_ab:
    eps = torch.randn(X0.shape, generator=rng, dtype=X0.dtype)
    Z_fwd.append(project(np.sqrt(ab) * X0 + np.sqrt(1 - ab) * eps))


all_z = np.concatenate([Z_data, *Z_rev, *Z_fwd])[:, :2]
lo, hi = np.percentile(all_z, [0.2, 99.8], axis=0)
pad = 0.05 * (hi - lo)
extent = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
gridsize = 45


def density(ax, z, cmap):
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
fig.savefig(out, dpi=150, transparent=True)
print(f"Saved {out}")
