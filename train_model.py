import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "trained-diffusion-model-fashion-mnist"  # read by sample_model.py and the visualizations

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

batch_size = 32

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(root=ROOT / "data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)

lr = 1e-4
total_steps = 100_000
log_every = 500
log_file = ROOT / "training_log.csv"

model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    block_out_channels=(64, 128, 128),
    layers_per_block=2,
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    dropout=0.1,
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

def infinite(loader):
    """Yield batches forever; the DataLoader reshuffles every time it starts a new pass over the data."""
    while True:
        for data, _ in loader:
            yield data

batches = infinite(train_loader)

log_f = open(log_file, "w", newline="")
log = csv.writer(log_f)
log.writerow(["step", "loss", "lr", "grad_norm", "steps_per_sec"])
loss_sum = torch.zeros((), device=device)
t0 = time.time()

model.train()
for step in range(1, total_steps + 1):
    imgs = next(batches).to(device)
    noise = torch.randn_like(imgs)
    t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

    noised_imgs = (
        sqrt_alphas_cumprod[t, None, None, None] * imgs
        + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
    )

    optimizer.zero_grad()
    estimated_noise = model(noised_imgs, t).sample
    loss = F.mse_loss(estimated_noise, noise)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.nn.utils.get_total_norm(grads)
    optimizer.step()
    loss_sum += loss.detach()

    if step % log_every == 0:
        loss_val, gnorm_val = (loss_sum / log_every).item(), grad_norm.item()
        loss_sum.zero_()
        lr_val = optimizer.param_groups[0]["lr"]
        steps_per_sec = log_every / (time.time() - t0)
        print(f"step {step:>7d} | loss {loss_val:.4f} | lr {lr_val:.2e} | gnorm {gnorm_val:.3f} | {steps_per_sec:.2f} steps/s")
        log.writerow([step, loss_val, lr_val, gnorm_val, steps_per_sec])
        log_f.flush()
        t0 = time.time()

log_f.close()

model.save_pretrained(MODEL_DIR)