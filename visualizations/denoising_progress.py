"""
Reverse process from pure noise to a sample, for a few generations, at many timesteps.

Longer version of figures/samples/denoising_progress.png (which sample_model.py makes with 10 frames):
it reads the trajectories saved by sample_trajectories.py, so no sampling is needed. Frame t of
trajectories.pt is x_t (t = 1000 is the initial noise, t = 0 the final sample).
Writes figures/samples/denoising_progress_long.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent  # repository root (this file is in visualizations/)
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
OUT_PATH = ROOT / "figures" / "samples" / "denoising_progress_long.png"
FRAMES = list(range(1000, -1, -40))  # columns: every 40 steps (must be multiples of the saved snapshot_every, 20)
SAMPLES = [3, 4, 6, 12, 14, 15, 44, 46]  # rows: trousers, sneaker, jacket, T-shirt, dress, boot, bag, shirt
INK = "#0b0b0b"

traj = torch.load(TRAJECTORIES_PATH, weights_only=False)
frame_t = traj["timesteps"].tolist()

fig, axes = plt.subplots(len(SAMPLES), len(FRAMES), figsize=(len(FRAMES) * 0.9, len(SAMPLES) * 0.9 + 0.5),
                         gridspec_kw={"wspace": 0.06, "hspace": 0.06})
for c, t in enumerate(FRAMES):
    fi = frame_t.index(t)
    for r, s in enumerate(SAMPLES):
        ax = axes[r, c]
        ax.imshow(((traj["x"][fi, s, 0].float().clamp(-1, 1) + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, c].set_title(f"t = {t}" if c == 0 else f"{t}", fontsize=11, color=INK, pad=4)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {OUT_PATH}")
