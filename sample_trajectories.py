"""
Draw many samples with `sample()` and save their raw reverse-diffusion trajectories x_t.

Output (`OUT_PATH`, a dict saved with torch.save):
    x               (n_frames, n_samples, 1, 28, 28) float16, raw x_t (no to_image clamp)
    timesteps       (n_frames,) t of each frame, from T (pure noise) down to 0 (final sample)
    alphas_cumprod  (n_frames,) alpha_bar of each frame (1.0 for t = 0)
    seed, snapshot_every

The file is rewritten after every batch, so an interrupted run still leaves the samples done so far.
"""

from pathlib import Path

import torch

from sample_model import alphas_cumprod, device, sample, timesteps

N_SAMPLES = 2000
BATCH_SIZE = 500
SNAPSHOT_EVERY = 20  # keep x_t every this many steps -> 51 frames for T = 1000
SEED = 0
OUT_PATH = Path("trajectories.pt")

# Frame 0 is x_T (pure noise). The snapshot taken after loop step t is x_t in 1-indexed DDPM
# notation, which was noised with alphas_cumprod[t - 1]; x_0 is the clean sample.
frame_ts = [timesteps] + [t for t in reversed(range(timesteps)) if t % SNAPSHOT_EVERY == 0]
frame_ab = [1.0 if t == 0 else alphas_cumprod[t - 1].item() for t in frame_ts]

generator = torch.Generator(device=device).manual_seed(SEED)
chunks = []
for start in range(0, N_SAMPLES, BATCH_SIZE):
    n = min(BATCH_SIZE, N_SAMPLES - start)
    print(f"Sampling {start}-{start + n} / {N_SAMPLES}")
    _, snapshots = sample(n, snapshot_every=SNAPSHOT_EVERY, generator=generator, verbose=False)
    chunks.append(torch.stack(snapshots).half())  # (n_frames, n, 1, 28, 28)

    torch.save(
        {
            "x": torch.cat(chunks, dim=1),
            "timesteps": torch.tensor(frame_ts),
            "alphas_cumprod": torch.tensor(frame_ab),
            "seed": SEED,
            "snapshot_every": SNAPSHOT_EVERY,
        },
        OUT_PATH,
    )
    print(f"Saved {start + n} trajectories to {OUT_PATH}")
