# Cell 1 — Imports & Hyperparameters
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

latent_dim     = 128
w_dim          = 128
batch_size     = 64
epochs         = 200
lr             = 0.0001
lambda_gp      = 10
n_critic       = 5
checkpoint_dir = "/content/checkpoints/stylegan_fixed"
image_dir      = "/content/images/stylegan_fixed"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset    = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Cell 2 — Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=latent_dim, w_dim=w_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2),
        )

    def forward(self, z):
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return self.net(z)

# Cell 3 — AdaIN & StyleBlock
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim=w_dim):
        super().__init__()
        self.norm        = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias  = nn.Linear(w_dim, channels)
        nn.init.ones_(self.style_scale.bias)
        nn.init.zeros_(self.style_bias.bias)

    def forward(self, x, w):
        x     = self.norm(x)
        gamma = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        beta  = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=w_dim, upsample=True):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            if upsample else nn.Identity()
        )
        self.conv  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adain = AdaIN(out_channels, w_dim)
        self.act   = nn.LeakyReLU(0.2)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, w)
        return self.act(x)

# Cell 4 — StyleGenerator
class StyleGenerator(nn.Module):
    def __init__(self, w_dim=w_dim):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, w_dim)
        self.const   = nn.Parameter(torch.randn(1, 256, 4, 4))
        self.block0  = StyleBlock(256, 128, w_dim, upsample=False)
        self.block1  = StyleBlock(128,  64, w_dim, upsample=True)
        self.block2  = StyleBlock( 64,  32, w_dim, upsample=True)
        self.block3  = StyleBlock( 32,  16, w_dim, upsample=True)
        self.to_rgb  = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z):
        b = z.size(0)
        w = self.mapping(z)
        x = self.const.expand(b, -1, -1, -1)   # (b, 256, 4, 4)
        x = self.block0(x, w)                   # (b, 128, 4, 4)
        x = self.block1(x, w)                   # (b,  64, 8, 8)
        x = self.block2(x, w)                   # (b,  32,16,16)
        x = self.block3(x, w)                   # (b,  16,32,32)
        x = x[:, :, 2:30, 2:30]                # centre-crop → (b,16,28,28)
        return self.to_rgb(x)                   # (b,   1,28,28)

# Discriminator spatial sizes for 28x28 input:
#   Conv1 (k=4,s=2,p=1): 28 -> 14
#   Conv2 (k=4,s=2,p=1): 14 ->  7
#   Conv3 (k=4,s=2,p=1):  7 ->  3
#   Conv4 (k=3,s=1,p=1):  3 ->  3
#   Flatten: 512 * 3 * 3 = 4608

class StyleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            # 7x7 -> 3x3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            # 3x3 -> 3x3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1)   # 4608 -> 1  (NO Sigmoid for WGAN)
        )

    def forward(self, img):
        return self.model(img)

# Cell 6 — Gradient Penalty
def gradient_penalty(D, real, fake, device):
    b     = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

    d_interp = D(interpolated)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.view(b, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

# Cell 7 — Instantiate Models & Optimisers
G = StyleGenerator(w_dim=w_dim).to(device)
D = StyleDiscriminator().to(device)

print(f"Generator parameters:     {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

# Cell 8 — Training Loop
z_fixed = torch.randn(16, latent_dim, device=device)

for epoch in range(1, epochs + 1):
    G.train()
    D.train()

    for step, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device, non_blocking=True)
        b = real_imgs.size(0)

        # ── Train Discriminator ──────────────────────────────────────────────
        z         = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z).detach()

        d_real = D(real_imgs)
        d_fake = D(fake_imgs)
        gp     = gradient_penalty(D, real_imgs, fake_imgs, device)

        loss_D = d_fake.mean() - d_real.mean() + lambda_gp * gp

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ── Train Generator (every n_critic steps) ───────────────────────────
        if step % n_critic == 0:
            z         = torch.randn(b, latent_dim, device=device)
            fake_imgs = G(z)
            loss_G    = -D(fake_imgs).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

    w_dist = d_real.mean().item() - d_fake.mean().item()
    print(f"Epoch {epoch}/{epochs}  LossD={loss_D.item():.4f}  LossG={loss_G.item():.4f}  W-dist={w_dist:.4f}")

    if epoch % 5 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "generator_state_dict":     G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizer_G_state_dict":   opt_G.state_dict(),
            "optimizer_D_state_dict":   opt_D.state_dict(),
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
        }, ckpt_path)
        print(f"  ✅ Checkpoint saved → {ckpt_path}")

        G.eval()
        with torch.no_grad():
            sample_imgs = G(z_fixed).cpu()
        G.train()

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(sample_imgs[idx][0], cmap='gray')
            ax.axis('off')
        plt.suptitle(f"StyleGAN (fixed) — Epoch {epoch}")
        plt.tight_layout()
        img_path = os.path.join(image_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(img_path, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  🖼️  Image saved   → {img_path}")

# Cell 9 — Final Visualisation
G.eval()
with torch.no_grad():
    z         = torch.randn(16, latent_dim, device=device)
    fake_imgs = G(z).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_imgs[i][0], cmap='gray')
    ax.axis('off')
plt.suptitle("StyleGAN (fixed) — Generated Images after 200 Epochs")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "final.png"), dpi=100, bbox_inches='tight')
plt.show()
print(f"All StyleGAN images saved in: {image_dir}")

# Cell 10 — Style Mixing Demo
G.eval()
with torch.no_grad():
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(8, latent_dim, device=device)

    w1 = G.mapping(z1).expand(8, -1)
    w2 = G.mapping(z2)

    x = G.const.expand(8, -1, -1, -1)
    x = G.block0(x, w1)
    x = G.block1(x, w1)
    x = G.block2(x, w2)
    x = G.block3(x, w2)
    x = x[:, :, 2:30, 2:30]
    mixed = G.to_rgb(x).cpu()

fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i, ax in enumerate(axes):
    ax.imshow(mixed[i][0], cmap='gray')
    ax.set_title(f"Mix {i+1}", fontsize=7)
    ax.axis('off')
plt.suptitle("Style Mixing — same structure (z1), different textures (z2)")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "style_mixing.png"), dpi=100, bbox_inches='tight')
plt.show()
print(f"Style mixing image saved → {image_dir}/style_mixing.png")

