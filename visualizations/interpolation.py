"""
Interpolation in the diffusion latent space (Ho, Jain & Abbeel 2020, Sec. 4.3) on Fashion-MNIST.

For two test images x_0, x_0':
  1. encode both to step t with the forward process, x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps,
     using the SAME eps for both (stochastic encoder);
  2. blend linearly, xbar_t = (1 - lam) x_t + lam x_t', lam = 0, 0.1, ..., 1;
  3. decode with the learned reverse process p(x_{t-1} | x_t) from t down to 0, with the SAME
     reverse noise for every lam, so the only thing that changes along a row is lam.

Because eps is shared, xbar_t equals the forward-noised pixel blend (1 - lam) x_0 + lam x_0': the
reverse process is what turns that ghostly superposition into a single plausible garment.
t trades fidelity for semantic mixing: small t stays close to the pixel blend, large t lets the
model invent more, and t = 1000 is (almost) pure noise, so the originals barely matter.

Timestep convention as in sample_trajectories.py: x_t (1-indexed) is noised with alphas_cumprod[t - 1]
and then decoded with loop steps t - 1, ..., 0 of the reverse process.

Results are cached in CACHE_PATH; delete it to recompute (~15-20 min on CPU). Writes
figures/analysis/interpolation_slide.png (a diagram of the process for one pair at T_MAIN, then more
examples).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent  # repository root (this file is in visualizations/)
sys.path.insert(0, str(ROOT))  # to import sample_model from the repository root

from sample_model import alphas, alphas_cumprod, betas, device, model

OUT_MAIN = ROOT / "figures" / "analysis" / "interpolation_slide.png"
CACHE_PATH = ROOT / "cache" / "interpolation.npz"
SEED = 0
T_MAIN = 250
LAMBDAS_MAIN = np.round(np.linspace(0, 1, 11), 2)  # steps of 0.1
FWD_STEPS = [0, 50, 100, 150, 200, 250]  # forward-process frames drawn in the diagram (last = T_MAIN)
REV_SNAPS = [125]  # reverse-process frames drawn between xbar_t and the result
LAM_DIAGRAM = 0.5
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# (class of x_0, class of x_0'): first test image of each class (the second one for x_0' when both
# classes are equal).
PAIRS = [(4, 2)]
# Extra examples drawn below the diagram: (class, n, class', n') = the n-th test image of each class.
EXTRA_PAIRS = [(1, 2, 1, 4), (7, 6, 9, 15), (5, 35, 9, 9), (4, 36, 2, 2)]
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

test = datasets.FashionMNIST(root=ROOT / "data", train=False, download=False,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,))]))
labels = test.targets.tolist()
of_class = {c: [i for i, y in enumerate(labels) if y == c] for c in range(10)}


def load_pairs(pairs):
    """(n_pairs, 1, 28, 28) tensors in [-1, 1] for x_0 and x_0'."""
    xa = torch.stack([test[of_class[a][0]][0] for a, _ in pairs])
    xb = torch.stack([test[of_class[b][int(a == b)]][0] for a, b in pairs])
    return xa, xb


def load_extra(pairs):
    xa = torch.stack([test[of_class[a][na]][0] for a, na, _, _ in pairs])
    xb = torch.stack([test[of_class[b][nb]][0] for _, _, b, nb in pairs])
    return xa, xb


x0_a, x0_b = load_pairs(PAIRS)


@torch.no_grad()
def interpolate(xa, xb, t, lambdas, snaps=()):
    """xa, xb: (P, 1, 28, 28). Returns decoded (P, L, 28, 28), the blended latents (P, L, 28, 28) and
    the reverse-process states x_s for s in snaps (n_snaps, P, L, 28, 28)."""
    g = torch.Generator().manual_seed(SEED)
    P, L = len(xa), len(lambdas)
    lam = torch.tensor(lambdas, dtype=torch.float32).view(1, L, 1, 1, 1)
    ab = alphas_cumprod[t - 1].cpu()
    eps = torch.randn((1, 1, 1, 28, 28), generator=g)  # shared by both images and all pairs
    xt_a = ab.sqrt() * xa[:, None] + (1 - ab).sqrt() * eps
    xt_b = ab.sqrt() * xb[:, None] + (1 - ab).sqrt() * eps
    x = ((1 - lam) * xt_a + lam * xt_b).reshape(P * L, 1, 28, 28)
    latent = x.clone()
    x = x.to(device)
    states = {}
    for i in reversed(range(t)):
        ti = torch.full((len(x),), i, device=device, dtype=torch.long)
        eps_hat = model(x, ti).sample
        z = torch.randn((1, 1, 28, 28), generator=g).to(device) if i > 0 else 0  # shared reverse noise
        x = (x - betas[i] / (1 - alphas_cumprod[i]).sqrt() * eps_hat) / alphas[i].sqrt() + betas[i].sqrt() * z
        if i in snaps:  # after loop step i the state is x_i
            states[i] = x.cpu().reshape(P, L, 28, 28).numpy()
        if i % 100 == 0:
            print(f"  t = {t}: step {i}")
    snapped = np.stack([states[s] for s in snaps]) if snaps else np.empty((0, P, L, 28, 28))
    return x.cpu().reshape(P, L, 28, 28).numpy(), latent.reshape(P, L, 28, 28).numpy(), snapped


def shared_eps():
    """The forward-process noise interpolate() draws first from its generator."""
    return torch.randn((28, 28), generator=torch.Generator().manual_seed(SEED)).numpy()


# Results are cached and recomputed only when their settings change.
c = dict(np.load(CACHE_PATH)) if CACHE_PATH.exists() else {}
main_key = dict(t_main=T_MAIN, pairs=np.array(PAIRS), lambdas_main=LAMBDAS_MAIN, rev_snaps=np.array(REV_SNAPS))


def cached(key):
    return all(k in c and np.array_equal(c[k], v) for k, v in key.items())


if cached(main_key):
    main, latent_main, rev_main = c["main"], c["latent_main"], c["rev_main"]
else:
    print(f"Using device: {device}")
    main, latent_main, rev_main = interpolate(x0_a, x0_b, T_MAIN, LAMBDAS_MAIN, REV_SNAPS)
    c.update(main_key, main=main, latent_main=latent_main, rev_main=rev_main)


def extra_key(q):  # one cache entry per extra pair, so the list can be edited without recomputing
    return f"extra_t{T_MAIN}_l{len(LAMBDAS_MAIN)}_" + "_".join(map(str, q))


missing = [q for q in EXTRA_PAIRS if extra_key(q) not in c]
if missing:
    print(f"Using device: {device}")
    for q, out in zip(missing, interpolate(*load_extra(missing), T_MAIN, LAMBDAS_MAIN)[0]):
        c[extra_key(q)] = out
extra = np.stack([c[extra_key(q)] for q in EXTRA_PAIRS]) if EXTRA_PAIRS else np.empty((0, len(LAMBDAS_MAIN), 28, 28))
ex_a, ex_b = load_extra(EXTRA_PAIRS) if EXTRA_PAIRS else (None, None)
CACHE_PATH.parent.mkdir(exist_ok=True)
np.savez(CACHE_PATH, **c)


def to_img(x):
    return (np.clip(x, -1, 1) + 1) / 2


def show(ax, img, edge=GRID, lw=1):
    ax.imshow(to_img(img), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(edge)
        sp.set_linewidth(lw)


# Figure 1 (PAIRS[0] only): the process for lam = LAM_DIAGRAM on top, the whole interpolation below.
#   row 0: forward process of x_0 (left -> centre) and of x_0' (right -> centre), shared eps
#   rows 1..: xbar_t = (1 - lam) x_t + lam x_t', then the reverse process going down column k_c
#   last row: x_0 | results for every lam | x_0', the lam = LAM_DIAGRAM result is the end of the column
ORANGE = "#eb6834"
L = len(LAMBDAS_MAIN)
n_cols = L + 2
k_c = 1 + int(np.argmin(np.abs(LAMBDAS_MAIN - LAM_DIAGRAM)))  # column of LAM_DIAGRAM
assert len(FWD_STEPS) <= k_c and FWD_STEPS[-1] == T_MAIN
xa, xb = x0_a[0, 0].numpy(), x0_b[0, 0].numpy()
eps = shared_eps()
ab = alphas_cumprod.cpu().numpy()


def forward(x0, s):
    return x0 if s == 0 else np.sqrt(ab[s - 1]) * x0 + np.sqrt(1 - ab[s - 1]) * eps


# Sizes in inches so that the diagram block and the examples block share the same columns.
W, LEFT, RIGHT, WS = 16.0, 0.03, 0.97, 0.3
cell = W * (RIGHT - LEFT) / (n_cols + (n_cols - 1) * WS)
HS_TOP, HS_EX = 0.55, 0.12
n_top = 3 + len(REV_SNAPS)  # forward, blend, reverse snapshots, results
n_ex = len(EXTRA_PAIRS)
top_h = cell * (n_top + (n_top - 1) * HS_TOP)
ex_h = cell * (n_ex + (n_ex - 1) * HS_EX) if n_ex else 0
TITLE, GAP, BOTTOM = 1.25, 0.8 if n_ex else 0, 0.2
H = TITLE + top_h + 0.45 + GAP + ex_h + BOTTOM
fig = plt.figure(figsize=(W, H))
y_top = 1 - TITLE / H
gs = fig.add_gridspec(n_top, n_cols, hspace=HS_TOP, wspace=WS, left=LEFT, right=RIGHT, top=y_top,
                      bottom=y_top - top_h / H)


def arrow(ax1, xy1, ax2, xy2, color=MUTED):
    fig.add_artist(ConnectionPatch(xyA=xy1, coordsA=ax1.transAxes, xyB=xy2, coordsB=ax2.transAxes,
                                   arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16, shrinkA=2,
                                   shrinkB=2))


def note(ax, x, y, step, text, ha="left", color=INK):
    """Step label next to an axis: a bold number and a short explanation."""
    ax.text(x, y, (rf"$\bf{{{step}}}$   " if step else "") + text, transform=ax.transAxes, ha=ha,
            va="center", fontsize=16, color=color, linespacing=1.5)


fwd_a, fwd_b = [], []
for j, s_ in enumerate(FWD_STEPS):
    fwd_a.append(fig.add_subplot(gs[0, k_c - len(FWD_STEPS) + j]))
    show(fwd_a[-1], forward(xa, s_))
    fwd_b.append(fig.add_subplot(gs[0, k_c + len(FWD_STEPS) - j]))
    show(fwd_b[-1], forward(xb, s_))
    if j:
        arrow(fwd_a[-2], (1, 0.5), fwd_a[-1], (0, 0.5))
        arrow(fwd_b[-2], (0, 0.5), fwd_b[-1], (1, 0.5))
fwd_a[0].set_title("$x_0$", fontsize=15, color=INK, pad=8)
fwd_b[0].set_title("$x_0'$", fontsize=15, color=INK, pad=8)
fwd_a[-1].set_title("$x_t$", fontsize=15, color=INK, pad=8)
fwd_b[-1].set_title("$x_t'$", fontsize=15, color=INK, pad=8)
centre_top = fig.add_subplot(gs[0, k_c])
centre_top.axis("off")
centre_top.text(0.5, 0.5, r"$\bf{1}$" + "\nadd noise", transform=centre_top.transAxes, ha="center",
                va="center", fontsize=16, color=INK, linespacing=1.4)

k = k_c - 1
lam_c = LAMBDAS_MAIN[k]
column = [fig.add_subplot(gs[1, k_c])]
show(column[0], latent_main[0, k], edge=ORANGE, lw=3)
arrow(fwd_a[-1], (0.5, 0), column[0], (0, 0.5))
arrow(fwd_b[-1], (0.5, 0), column[0], (1, 0.5))
note(column[0], 1.3, 0.5, 2, f"mix  (λ = {lam_c:g})")
for r, s_ in enumerate(REV_SNAPS):
    column.append(fig.add_subplot(gs[2 + r, k_c]))
    show(column[-1], rev_main[r, 0, k], edge=BLUE, lw=2)
note(column[1], 1.3, 0.5, 3, "denoise")

ax = fig.add_subplot(gs[n_top - 1, 0])
show(ax, xa)
ax.set_xlabel("$x_0$", fontsize=14, color=INK)
ax = fig.add_subplot(gs[n_top - 1, -1])
show(ax, xb)
ax.set_xlabel("$x_0'$", fontsize=14, color=INK)
for kk, lam in enumerate(LAMBDAS_MAIN):
    ax = fig.add_subplot(gs[n_top - 1, kk + 1])
    centre = kk + 1 == k_c
    show(ax, main[0, kk], edge=ORANGE if centre else BLUE, lw=3 if centre else 2)
    ax.set_xlabel(f"λ = {lam:g}" if kk == 0 else f"{lam:g}", fontsize=13, color=INK,
                  fontweight="bold" if centre else "normal")
    if centre:
        column.append(ax)
for top, bottom in zip(column[:-1], column[1:]):
    arrow(top, (0.5, 0), bottom, (0.5, 1))
fig.suptitle(f"Interpolation in the diffusion latent space (t = {T_MAIN})", fontsize=24,
             color=INK, y=1 - 0.35 / H)

if n_ex:
    y_line = (BOTTOM + ex_h + GAP * 0.5) / H
    fig.add_artist(Line2D([LEFT - 0.05, RIGHT + 0.05], [y_line, y_line], color=GRID, lw=2))
    gs_ex = fig.add_gridspec(n_ex, n_cols, hspace=HS_EX, wspace=WS, left=LEFT, right=RIGHT,
                             top=(BOTTOM + ex_h) / H, bottom=BOTTOM / H)
    for r, (a, _, b, _) in enumerate(EXTRA_PAIRS):
        ax = fig.add_subplot(gs_ex[r, 0])
        show(ax, ex_a[r, 0].numpy())
        ax = fig.add_subplot(gs_ex[r, -1])
        show(ax, ex_b[r, 0].numpy())
        for kk in range(L):
            show(fig.add_subplot(gs_ex[r, kk + 1]), extra[r, kk], edge=BLUE, lw=2)
OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_MAIN, dpi=150, transparent=True)
print(f"Saved {OUT_MAIN}")
