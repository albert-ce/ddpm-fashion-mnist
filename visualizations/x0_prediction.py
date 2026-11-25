from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import UNet2DModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "trained-diffusion-model-fashion-mnist"
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
OUT_PATH = ROOT / "figures" / "analysis" / "x0_prediction_slide.png"
CACHE_PATH = ROOT / "cache" / "x0_prediction.npz"
N_EVAL = 200
BATCH_SIZE = 200
FRAMES = [1000, 700, 500, 300, 240, 200, 100, 20]
SAMPLES = [0, 1, 2]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

traj = torch.load(TRAJECTORIES_PATH, weights_only=False)
frame_t = traj["timesteps"].tolist()  # 1000, 980, ..., 20, 0
final = traj["x"][frame_t.index(0), :N_EVAL].float()


def compute():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    alphas_cumprod = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, 1000), dim=0)
    model = UNet2DModel.from_pretrained(MODEL_DIR).to(device).eval()

    ts = [t for t in frame_t if t > 0]
    err = np.empty(len(ts))
    err_noclip = np.empty(len(ts))
    shown = np.empty((len(ts), len(SAMPLES), 28, 28), dtype=np.float32)
    with torch.no_grad():
        for i, t in enumerate(ts):
            ab = alphas_cumprod[t - 1]
            x_t = traj["x"][frame_t.index(t), :N_EVAL].float()
            x0_hat = torch.empty_like(x_t)
            for b in range(0, N_EVAL, BATCH_SIZE):
                xb = x_t[b: b + BATCH_SIZE].to(device)
                tb = torch.full((len(xb),), t - 1, device=device, dtype=torch.long)
                eps_hat = model(xb, tb).sample.cpu()
                x0_hat[b: b + BATCH_SIZE] = (xb.cpu() - (1 - ab).sqrt() * eps_hat) / ab.sqrt()
            err_noclip[i] = (x0_hat - final).pow(2).mean().item()
            clipped = x0_hat.clamp(-1, 1)
            err[i] = (clipped - final).pow(2).mean().item()
            shown[i] = clipped[SAMPLES, 0].numpy()
            print(f"t = {t:4d} | MSE(x0_hat, final) clipped {err[i]:.4f} | unclipped {err_noclip[i]:.4f}")
    return np.array(ts), err, err_noclip, shown


if CACHE_PATH.exists():
    cache = np.load(CACHE_PATH)
    ts, err, err_noclip, shown = cache["t"], cache["err"], cache["err_noclip"], cache["shown"]
    print(f"Loaded {CACHE_PATH}")
else:
    ts, err, err_noclip, shown = compute()
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez(CACHE_PATH, t=ts, err=err, err_noclip=err_noclip, shown=shown)

blind = (final.mean(dim=0, keepdim=True) - final).pow(2).mean().item()
ts_list = ts.tolist()


def to_img(x):
    return (np.clip(x, -1, 1) + 1) / 2


# Figure
n_cols = len(FRAMES) + 1
fig = plt.figure(figsize=(16, 15.5))
gs = fig.add_gridspec(1 + 2 * len(SAMPLES), n_cols,
                      height_ratios=[3.6] + [1] * (2 * len(SAMPLES)),
                      hspace=0.1, wspace=0.08, left=0.1, right=0.98, top=0.9, bottom=0.02)

ax = fig.add_subplot(gs[0, :])
ax.axhline(blind, color=MUTED, lw=1.5, ls="--")
ax.text(15, blind * 1.12, "no information (predict the mean image)", color=MUTED, ha="right", va="bottom",
        fontsize=14)
ax.plot(ts, err, color=BLUE, lw=4)
for k, f in enumerate(FRAMES):
    y = err[ts_list.index(f)]
    ax.plot(f, y, "o", color=BLUE, markersize=13, markeredgecolor="white", markeredgewidth=2.5, zorder=3)
    ax.annotate(chr(ord("A") + k), (f, y), xytext=(0, 14), textcoords="offset points", ha="center",
                va="bottom", fontsize=17, fontweight="bold", color=INK)
ax.set_yscale("log")
ax.set_ylim(err.min() / 3, max(err.max(), blind) * 3)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
ax.set_xlim(1010, -10)  # generation order
ax.set_ylabel(r"MSE( predicted $\hat x_0$ , final image )", fontsize=17)
ax.set_xlabel("diffusion timestep t   (generation goes left → right)", fontsize=15, labelpad=6)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.grid(color=GRID, lw=0.8)
ax.spines[["bottom", "right"]].set_visible(False)
ax.tick_params(colors=MUTED, which="both", labelsize=14)
fig.suptitle("When does the model know which image it will generate?", fontsize=24, color=INK, y=0.975)

for r, s in enumerate(SAMPLES):
    for k, f in enumerate(FRAMES + [0]):
        for row, kind in enumerate(["x_t", "x0"]):
            ax_i = fig.add_subplot(gs[1 + 2 * r + row, k])
            if f == 0:
                img = to_img(final[s, 0].numpy())
            elif kind == "x_t":
                img = to_img(traj["x"][frame_t.index(f), s, 0].float().numpy())
            else:
                img = to_img(shown[ts_list.index(f), r])
            ax_i.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax_i.set_xticks([])
            ax_i.set_yticks([])
            is_pred = kind == "x0" and f != 0
            for sp in ax_i.spines.values():
                sp.set_edgecolor(BLUE if is_pred else GRID)
                sp.set_linewidth(2.5 if is_pred else 1)
            if r == 0 and row == 0:
                label = "final" if f == 0 else f"{chr(ord('A') + k)}   " + ("noise" if f == 1000 else f"t = {f}")
                ax_i.set_title(label, fontsize=15, fontweight="bold", color=INK, pad=6)
            if k == 0:
                ax_i.set_ylabel(r"$x_t$" if kind == "x_t" else r"$\hat x_0$", fontsize=18, rotation=0,
                                labelpad=22, va="center", color=BLUE if kind == "x0" else INK)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, transparent=True)
print(f"Saved {OUT_PATH}")
