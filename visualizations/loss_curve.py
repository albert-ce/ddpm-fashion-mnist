"""
Training loss vs. step, from training_log.csv (written by train_model.py).

Each row's loss is the mean over the logging window (500 steps x 32 images), so it is plotted as it
is, with no extra smoothing. The y axis is logarithmic so that both the fast initial drop and the
slow decline afterwards are visible in one panel. Writes figures/training/loss_curve.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # repository root (this file is in visualizations/)
LOG_PATH = ROOT / "training_log.csv"
OUT_PATH = ROOT / "figures" / "training" / "loss_curve.png"
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

log = np.genfromtxt(LOG_PATH, delimiter=",", names=True)
step, loss = log["step"], log["loss"]
window = int(step[1] - step[0])

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(step, loss, color=BLUE, lw=2.5)
for i, ha, dx in [(0, "left", 8), (len(step) - 1, "right", 0)]:
    ax.plot(step[i], loss[i], "o", color=BLUE, markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.annotate(f"{loss[i]:.3f}", (step[i], loss[i]), xytext=(dx, 10), textcoords="offset points", ha=ha,
                va="bottom", fontsize=13, color=INK)
ax.set_yscale("log")
ax.set_yticks([0.03, 0.04, 0.05, 0.07, 0.1], labels=["0.03", "0.04", "0.05", "0.07", "0.1"])
ax.minorticks_off()
ax.set_ylim(0.028, 0.125)
ax.set_xlim(0, step[-1] * 1.02)
ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k")
ax.set_xlabel("training step", fontsize=14)
ax.set_ylabel(f"MSE loss on ε  (mean of {window} steps)", fontsize=14)
ax.set_title("Training loss", fontsize=17, color=INK, pad=12)
ax.grid(color=GRID, lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=12)
fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, transparent=True)
print(f"Saved {OUT_PATH}")
