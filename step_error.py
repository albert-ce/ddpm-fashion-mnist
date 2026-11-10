"""
Error of a single reverse (generation) step x_t -> x_{t-1}, relative to the size of the image it builds.
Writes training_figures/step_error_slide.png.

One generation step is x_{t-1} = mu + sigma_t z. If x_0 were known, the ideal mean would be
    mu_tilde = 1/sqrt(a_t) (x_t - b_t / sqrt(1 - ab_t) eps)       (true noise eps)
and the model uses
    mu_theta = 1/sqrt(a_t) (x_t - b_t / sqrt(1 - ab_t) eps_hat)   (predicted noise eps_hat)
so the error of the step is mu_theta - mu_tilde = b_t / sqrt(a_t (1 - ab_t)) (eps - eps_hat).

For each t, the same N_IMAGES test images are noised to x_t and the per-pixel RMS of that error is
computed (sigma_t = sqrt(b_t), the std of the noise the step adds, is also cached).

t = 0 (the last step, which adds no noise) is left out. The numbers are cached in CACHE_PATH; delete it
to recompute (e.g. after changing N_IMAGES, T_VALUES or the model).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import UNet2DModel
from torchvision import datasets, transforms

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "trained-diffusion-model-fashion-mnist"
CACHE_PATH = ROOT / "training_figures" / "step_error.npz"
N_IMAGES = 200  # test images per t (~44 images/s on CPU)
# Dense at small t, where the error is closest to the noise.
T_VALUES = list(range(1, 20)) + list(range(20, 1000, 20)) + [999]
SEED = 0
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"



def compute():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    model = UNet2DModel.from_pretrained(MODEL_DIR).to(device).eval()

    test_set = datasets.FashionMNIST(
        root=ROOT / "data", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )
    generator = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(test_set), generator=generator)[:N_IMAGES]
    x0 = torch.stack([test_set[i][0] for i in idx.tolist()]).to(device)

    step_err = np.empty(len(T_VALUES))  # per-pixel RMS of mu_theta - mu_tilde
    sigma = np.empty(len(T_VALUES))  # std of the noise added by the step
    with torch.no_grad():
        for i, t in enumerate(T_VALUES):
            eps = torch.randn(x0.shape, generator=generator).to(device)
            ab, a, b = alphas_cumprod[t], alphas[t], betas[t]
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            eps_pred = model(x_t, torch.full((N_IMAGES,), t, device=device, dtype=torch.long)).sample
            coef = b / (a * (1 - ab)).sqrt()
            step_err[i] = (coef * (eps_pred - eps)).pow(2).mean().sqrt().item()
            sigma[i] = b.sqrt().item()
            print(f"t = {t:4d} | step error {step_err[i]:.5f} | sigma {sigma[i]:.5f} | ratio {step_err[i] / sigma[i]:.3f}")
    return np.array(T_VALUES), step_err, sigma


if CACHE_PATH.exists():
    cache = np.load(CACHE_PATH)
    T_VALUES, step_err, sigma = cache["t"].tolist(), cache["step_err"], cache["sigma"]
    print(f"Loaded {CACHE_PATH}")
else:
    t_arr, step_err, sigma = compute()
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez(CACHE_PATH, t=t_arr, step_err=step_err, sigma=sigma)

# The forward process shrinks the image: x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps, so x_{t-1} (the
# output of the step, whose mean is being compared) contains the image scaled by sqrt(ab_{t-1}).
# Generation runs that backwards, so the image grows from ~0.006 to 1 and the absolute error grows
# with it. Dividing by sqrt(ab_{t-1}) removes that scale: if it were the only cause, the ratio
# would be flat.
t_np = np.array(T_VALUES)
ab_np = np.cumprod(1.0 - np.linspace(1e-4, 0.02, 1000))
image_scale = np.sqrt(ab_np[t_np - 1])  # sqrt(ab_{t-1})
rel_err = step_err / image_scale

# Figure: absolute step error vs. image size (top), their ratio (middle) and real generated x_t at
# marked t (bottom), taken from the trajectories saved by sample_trajectories.py. Frame "t" of that
# file is x_t in 1-indexed DDPM notation (noised with alphas_cumprod[t - 1]), one step off the
# 0-indexed t of the curves; at this scale that makes no visible difference.
TRAJECTORIES_PATH = ROOT / "trajectories.pt"
OUT_PATH_SLIDE = ROOT / "training_figures" / "step_error_slide.png"
FRAMES = [1000, 700, 500, 300, 240, 200, 100, 0]  # must be multiples of the saved snapshot_every
SAMPLES = [0, 1, 2]  # which trajectories to show (one row each)
MIN_BAND = (260, 180)  # t range highlighted around the minimum

if TRAJECTORIES_PATH.exists():
    traj = torch.load(TRAJECTORIES_PATH, weights_only=False)
    frame_t = traj["timesteps"].tolist()
    t_curve, rel_curve = t_np, rel_err  # T_VALUES is increasing, as np.interp needs

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2 + len(SAMPLES), len(FRAMES), height_ratios=[2.6, 3.0] + [1] * len(SAMPLES),
                          hspace=0.14, wspace=0.08, left=0.1, right=0.98, top=0.88, bottom=0.03)

    # Top: absolute step error and the size of the image inside x_{t-1}.
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axvspan(*MIN_BAND, color=BLUE, alpha=0.08, linewidth=0)
    ax_top.plot(t_np, image_scale, color=ORANGE, lw=4)
    ax_top.plot(t_np, step_err, color=BLUE, lw=4)
    ax_top.text(520, 0.035, r"size of the image inside $x_{t-1}$:  $\sqrt{\bar\alpha_{t-1}}$",
                color=ORANGE, ha="center", va="top", fontsize=17, fontweight="bold")
    ax_top.text(560, 0.0017, r"model error in the step:  $\|\mu_\theta - \tilde\mu\|$",
                color=BLUE, ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("per-pixel size", fontsize=17)
    ax_top.set_xlabel("diffusion timestep t   (generation goes left → right)", fontsize=15, labelpad=6)
    ax_top.xaxis.set_label_position("top")
    ax_top.xaxis.tick_top()
    ax_top.spines[["bottom", "right"]].set_visible(False)

    # Middle: their ratio, with the frames shown below marked.
    ax = fig.add_subplot(gs[1, :], sharex=ax_top)
    ax.axvspan(*MIN_BAND, color=BLUE, alpha=0.08, linewidth=0)
    ax.plot(t_np, rel_err, color=BLUE, lw=4)
    for k, f in enumerate(FRAMES):
        x = max(f, 1)
        y = np.interp(x, t_curve, rel_curve)
        ax.plot(x, y, "o", color=BLUE, markersize=13, markeredgecolor="white", markeredgewidth=2.5, zorder=3)
        ax.annotate(chr(ord("A") + k), (x, y), xytext=(0, 14), textcoords="offset points", ha="center",
                    va="bottom", fontsize=17, fontweight="bold", color=INK)
    j_min = int(np.argmin(rel_err))
    ax.annotate(f"minimum\nt ≈ {T_VALUES[j_min]}", (T_VALUES[j_min], rel_err[j_min]), xytext=(0, -18),
                textcoords="offset points", ha="center", va="top", fontsize=15, color=BLUE, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(rel_err.min() / 3, rel_err.max() * 3)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.set_xlim(1010, -10)  # generation order (shared with ax_top)
    ax.set_ylabel("error / image size", fontsize=17)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.spines[["bottom", "top", "right"]].set_visible(False)
    for a_ in (ax_top, ax):
        a_.grid(color=GRID, lw=0.8)
        a_.tick_params(colors=MUTED, which="both", labelsize=14)
    fig.suptitle("Model error in each generation step, relative to the image it is building",
                 fontsize=24, color=INK, y=0.975)

    for k, f in enumerate(FRAMES):
        fi = frame_t.index(f)
        for r, s in enumerate(SAMPLES):
            ax_i = fig.add_subplot(gs[2 + r, k])
            img = (traj["x"][fi, s, 0].float().clamp(-1, 1) + 1) / 2
            ax_i.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
            ax_i.set_xticks([])
            ax_i.set_yticks([])
            in_band = MIN_BAND[1] <= f <= MIN_BAND[0]
            for sp in ax_i.spines.values():
                sp.set_edgecolor(BLUE if in_band else GRID)
                sp.set_linewidth(3 if in_band else 1)
            if r == 0:
                label = f"{chr(ord('A') + k)}   " + ("noise" if f == 1000 else "final" if f == 0 else f"t = {f}")
                ax_i.set_title(label, fontsize=15, fontweight="bold", color=BLUE if in_band else INK, pad=6)
    fig.savefig(OUT_PATH_SLIDE, dpi=150)
    print(f"Saved {OUT_PATH_SLIDE}")
else:
    print(f"{TRAJECTORIES_PATH} not found (run sample_trajectories.py): skipping {OUT_PATH_SLIDE.name}")
