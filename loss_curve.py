"""
Training loss vs. step, from training_log.csv (written by train_model.py).

Each row's loss is the mean over the LOG_EVERY steps of its window (500 steps x 32 images), so the
values are smooth enough to plot as they are, with no extra smoothing.

Left: the whole run, where the fast initial drop dominates. Right: zoom from ZOOM_FROM steps on, with
its own y range, to show that the loss keeps going down slowly until the end.
Writes training_figures/loss_curve.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
LOG_PATH = ROOT / "training_log.csv"
OUT_PATH = ROOT / "training_figures" / "loss_curve.png"
ZOOM_FROM = 5000  # first step of the right panel
STEPS_PER_EPOCH = 60_000 // 32  # FashionMNIST train set / batch size (drop_last)
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

log = np.genfromtxt(LOG_PATH, delimiter=",", names=True)
step, loss = log["step"], log["loss"]
lr = log["lr"]
lr_str = f"{lr[0]:.0e}" if np.all(lr == lr[0]) else "variable"
zoom = step >= ZOOM_FROM


def label(ax, i, text_offset, ha, va="bottom"):
    ax.plot(step[i], loss[i], "o", color=BLUE, markersize=11, markeredgecolor="white", markeredgewidth=2.5,
            zorder=3)
    ax.annotate(f"{loss[i]:.4f}\n(step {step[i] / 1000:g}k)", (step[i], loss[i]), xytext=text_offset,
                textcoords="offset points", ha=ha, va=va, fontsize=14, fontweight="bold", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))


fig, (ax, ax_z) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})

# Left: whole run
ax.plot(step, loss, color=BLUE, lw=3.5)
ax.axvspan(ZOOM_FROM, step[-1], color=BLUE, alpha=0.06, linewidth=0)
ax.text((ZOOM_FROM + step[-1]) / 2, loss.max() * 0.98, "zoom →", ha="center", va="top", fontsize=14,
        color=MUTED)
label(ax, 0, (10, 0), "left")
label(ax, len(step) - 1, (0, 12), "right")
ax.set_ylim(0, loss.max() * 1.12)
ax.set_title("Whole training run", fontsize=18, color=INK, pad=10)

# Right: zoom
ax_z.plot(step[zoom], loss[zoom], color=BLUE, lw=3)
i_zoom = int(np.argmax(zoom))
label(ax_z, i_zoom, (10, 6), "left")
label(ax_z, len(step) - 1, (0, -18), "right", va="top")
pad = (loss[zoom].max() - loss[zoom].min()) * 0.25
ax_z.set_ylim(loss[zoom].min() - pad, loss[zoom].max() + pad)
drop = (1 - loss[-1] / loss[i_zoom]) * 100
ax_z.set_title(f"From step {ZOOM_FROM / 1000:g}k on: still {drop:.0f}% lower by the end", fontsize=18, color=INK,
               pad=10)

for a in (ax, ax_z):
    a.set_xlabel("training step", fontsize=16)
    a.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k")
    a.set_xlim(0 if a is ax else ZOOM_FROM - 3000, step[-1] * 1.02)
    a.grid(color=GRID, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(colors=MUTED, labelsize=14)
    sec = a.secondary_xaxis("top", functions=(lambda s: s / STEPS_PER_EPOCH, lambda e: e * STEPS_PER_EPOCH))
    sec.set_xlabel("epoch", fontsize=14, color=MUTED)
    sec.tick_params(colors=MUTED, labelsize=12)
    sec.spines["top"].set_visible(False)
ax.set_ylabel(f"MSE loss on ε  (mean over {int(step[0])} steps)", fontsize=16)

fig.suptitle(f"Training loss  (Adam, lr {lr_str}, batch 32, {int(step[-1]) // 1000}k steps)", fontsize=22,
             color=INK, y=1.13)
OUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
