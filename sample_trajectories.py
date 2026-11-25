from pathlib import Path

import torch

from sample_model import alphas_cumprod, device, sample, timesteps

N_SAMPLES = 2000
BATCH_SIZE = 500
SNAPSHOT_EVERY = 20
SEED = 0
OUT_PATH = Path(__file__).resolve().parent / "trajectories.pt"

frame_ts = [timesteps] + [t for t in reversed(range(timesteps)) if t % SNAPSHOT_EVERY == 0]
frame_ab = [1.0 if t == 0 else alphas_cumprod[t - 1].item() for t in frame_ts]

if __name__ == "__main__":
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
