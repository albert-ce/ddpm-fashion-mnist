from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
OUT_PATH = ROOT / "figures" / "samples" / "denoising_progress.png"
FRAMES = [1000, 600, 400, 300, 240, 200, 160, 120, 80, 60, 40, 20, 0]
SAMPLES = [3, 4, 6, 14, 15, 44]  # trousers, sneaker, jacket, dress, boot, bag
LABEL_SIZE = 20
INK = "#0b0b0b"

traj = torch.load(TRAJECTORIES_PATH, weights_only=False)
frame_t = traj["timesteps"].tolist()

CELL = 1.3  # inches per image
fig, axes = plt.subplots(len(SAMPLES), len(FRAMES), figsize=(len(FRAMES) * CELL, len(SAMPLES) * CELL + 0.8),
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
    axes[0, c].set_title(f"{t}", fontsize=LABEL_SIZE, fontweight="bold", color=INK, pad=8)
axes[0, 0].annotate("t", (0, 1), xycoords="axes fraction", xytext=(-14, 8), textcoords="offset points",
                    ha="right", va="bottom", fontsize=LABEL_SIZE + 8, fontweight="bold", fontstyle="italic",
                    color=INK)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {OUT_PATH}")
