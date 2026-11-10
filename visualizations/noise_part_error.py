"""
Training loss per timestep t, and the same error measured in the units of x_t.

The training loss is the MSE on the noise eps. But eps enters x_t scaled:
    x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps,
so at small t the noise is a tiny part of x_t and any small error is blown up when measured on eps.
Multiplying by (1 - ab_t) measures the error of the noise PART of x_t instead:
    (1 - ab_t) MSE(eps_hat, eps) = MSE( sqrt(1 - ab_t) eps_hat , sqrt(1 - ab_t) eps ),
which, since x_t is fixed, is also the error of the image part, ab_t MSE(x0_hat, x0): it says how
well the model splits what it sees into image and noise, whichever of the two you look at.

Top: loss on eps per t and the noise share 1 - ab_t. Middle: their product, with marked steps.
Bottom: real generated x_t at those steps (trajectories.pt, from sample_trajectories.py; its frame t
is noised with alphas_cumprod[t - 1], one step off the 0-indexed t of the curves, invisible here).

For each t the same N_IMAGES test images are noised to x_t. The numbers are cached in CACHE_PATH;
delete it to recompute. Writes figures/analysis/noise_part_error_slide.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import UNet2DModel
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent  # repository root (this file is in visualizations/)
MODEL_DIR = ROOT / "trained-diffusion-model-fashion-mnist"
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
OUT_PATH = ROOT / "figures" / "analysis" / "noise_part_error_slide.png"
CACHE_PATH = ROOT / "cache" / "noise_part_error.npz"
N_IMAGES = 200  # test images per t (~44 images/s on CPU)
T_VALUES = list(range(0, 20, 2)) + list(range(20, 1000, 20)) + [999]
SEED = 0
FRAMES = [1000, 700, 500, 400, 260, 200, 100, 20]  # columns A-H (multiples of the saved snapshot_every)
SAMPLES = [44, 45, 46]  # trajectories shown (one row each): a bag, a sneaker, a shirt
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

alphas_cumprod = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, 1000), dim=0)


def compute():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = UNet2DModel.from_pretrained(MODEL_DIR).to(device).eval()
    test_set = datasets.FashionMNIST(
        root=ROOT / "data", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )
    generator = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(test_set), generator=generator)[:N_IMAGES]
    x0 = torch.stack([test_set[i][0] for i in idx.tolist()]).to(device)

    mse_eps = np.empty(len(T_VALUES))
    with torch.no_grad():
        for i, t in enumerate(T_VALUES):
            ab = alphas_cumprod[t].to(device)
            eps = torch.randn(x0.shape, generator=generator).to(device)
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            eps_hat = model(x_t, torch.full((N_IMAGES,), t, device=device, dtype=torch.long)).sample
            mse_eps[i] = (eps_hat - eps).pow(2).mean().item()
            print(f"t = {t:4d} | MSE eps {mse_eps[i]:.5f} | noise part {(1 - ab.item()) * mse_eps[i]:.5f}")
    return np.array(T_VALUES), mse_eps


if CACHE_PATH.exists():
    cache = np.load(CACHE_PATH)
    t_np, mse_eps = cache["t"], cache["mse_eps"]
    print(f"Loaded {CACHE_PATH}")
else:
    t_np, mse_eps = compute()
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez(CACHE_PATH, t=t_np, mse_eps=mse_eps)

noise_share = 1 - alphas_cumprod.numpy()[t_np]
part_err = noise_share * mse_eps

traj = torch.load(TRAJECTORIES_PATH, weights_only=False)
frame_t = traj["timesteps"].tolist()

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(2 + len(SAMPLES), len(FRAMES), height_ratios=[2.6, 3.0] + [1] * len(SAMPLES),
                      hspace=0.14, wspace=0.08, left=0.1, right=0.98, top=0.88, bottom=0.03)

# Top: the training loss per t and the share of x_t's variance that is noise.
ax_top = fig.add_subplot(gs[0, :])
ax_top.plot(t_np, noise_share, color=ORANGE, lw=4)
ax_top.plot(t_np, mse_eps, color=BLUE, lw=4)
ax_top.text(560, 0.35, r"noise share of $x_t$:  $1-\bar\alpha_t$", color=ORANGE, ha="center", va="top",
            fontsize=17, fontweight="bold")
ax_top.text(330, 0.0012, r"training loss at each t:  MSE$(\hat\varepsilon, \varepsilon)$", color=BLUE,
            ha="center", va="top", fontsize=17, fontweight="bold")
ax_top.set_yscale("log")
ax_top.set_ylabel("per-pixel value", fontsize=17)
ax_top.set_xlabel("diffusion timestep t   (generation goes left → right)", fontsize=15, labelpad=6)
ax_top.xaxis.set_label_position("top")
ax_top.xaxis.tick_top()
ax_top.spines[["bottom", "right"]].set_visible(False)

# Middle: their product = error of the noise part of x_t (= error of the image part).
ax = fig.add_subplot(gs[1, :], sharex=ax_top)
ax.plot(t_np, part_err, color=BLUE, lw=4)
for k, f in enumerate(FRAMES):
    x = min(f, 999)
    y = np.interp(x, t_np, part_err)
    ax.plot(x, y, "o", color=BLUE, markersize=13, markeredgecolor="white", markeredgewidth=2.5, zorder=3)
    ax.annotate(chr(ord("A") + k), (x, y), xytext=(0, 14), textcoords="offset points", ha="center",
                va="bottom", fontsize=17, fontweight="bold", color=INK)
ax.set_ylim(0, part_err.max() * 1.25)
ax.set_xlim(1010, -10)  # generation order (shared with ax_top)
ax.set_ylabel(r"error of the noise part" + "\n" + r"$(1-\bar\alpha_t)\,$MSE$(\hat\varepsilon, \varepsilon)$",
              fontsize=17)
ax.tick_params(axis="x", labelbottom=False, bottom=False)
ax.spines[["bottom", "top", "right"]].set_visible(False)
for a_ in (ax_top, ax):
    a_.grid(color=GRID, lw=0.8)
    a_.tick_params(colors=MUTED, which="both", labelsize=14)
fig.suptitle("Training loss per t, and the same error in the units of $x_t$", fontsize=24, color=INK, y=0.975)

for k, f in enumerate(FRAMES):
    fi = frame_t.index(f)
    for r, s in enumerate(SAMPLES):
        ax_i = fig.add_subplot(gs[2 + r, k])
        img = (traj["x"][fi, s, 0].float().clamp(-1, 1) + 1) / 2
        ax_i.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        for sp in ax_i.spines.values():
            sp.set_edgecolor(GRID)
        if r == 0:
            label = f"{chr(ord('A') + k)}   " + ("noise" if f == 1000 else f"t = {f}")
            ax_i.set_title(label, fontsize=15, fontweight="bold", color=INK, pad=6)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, transparent=True)
print(f"Saved {OUT_PATH}")
