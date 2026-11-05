import torch
from diffusers import UNet2DModel
from pathlib import Path
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_dir = Path("./trained-diffusion-model-fashion-mnist")
num_samples = 64
seed = 0
grid_path = "samples_grid.png"
progress_path = "samples_progress.png"
num_progress_samples = 8
num_progress_frames = 10

beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

model = UNet2DModel.from_pretrained(model_dir).to(device)
model.eval()


def to_image(x):
    # Undo Normalize((0.5,), (0.5,)): [-1, 1] -> [0, 1]
    return (x.clamp(-1, 1) + 1) / 2

@torch.no_grad()
def sample(n_samples, snapshot_every=None, n_progress_samples=None, generator=None, verbose=True):
    x = torch.randn((n_samples, 1, 28, 28), device=device, generator=generator)
    snapshots = []
    if snapshot_every is not None:
        n_progress_samples = n_progress_samples or n_samples
        snapshots.append(x[:n_progress_samples].cpu())

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_tensor).sample

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        # We add noise to make x_{t-1} a true sample of N(mu, beta_tilde_t·I), keeping its variance consistent.
        if t > 0:
            noise = torch.randn(x.shape, device=device, generator=generator)
        else:
            noise = 0

        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta) + sqrt(beta_t) * noise
        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise

        if snapshot_every is not None and (t % snapshot_every == 0):
            snapshots.append(x[:n_progress_samples].cpu())

        if verbose and t % 100 == 0:
            print(f"t = {t}")

    return x, snapshots


if __name__ == "__main__":
    generator = torch.Generator(device=device).manual_seed(seed)
    samples, snapshots = sample(num_samples, snapshot_every=timesteps // num_progress_frames, n_progress_samples=num_progress_samples, generator=generator)

    save_image(to_image(samples), grid_path, nrow=8)
    print(f"Saved {num_samples} samples to {grid_path}")

    progress = to_image(torch.stack(snapshots, dim=1)) # (n_samples, n_frames, 1, 28, 28)
    progress = progress.reshape(-1, 1, 28, 28)
    save_image(progress, progress_path, nrow=len(snapshots))
    print(f"Saved denoising progress to {progress_path}")
