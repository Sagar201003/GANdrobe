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

latent_dim = 100
batch_size = 128        # larger batch → better GPU utilisation on T4
lr = 0.0002
epochs = 300
checkpoint_dir = "/content/checkpoints/vanilla_gan"
image_dir      = "/content/images/vanilla_gan"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,      # T4 Colab has 2 CPU cores available
    pin_memory=True     # faster CPU→GPU transfers
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(1, epochs + 1):
    G.train()
    D.train()
    for i, (real_imgs, _) in enumerate(dataloader):

        real_imgs = real_imgs.view(-1, 784).to(device, non_blocking=True)
        batch_size_curr = real_imgs.size(0)

        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # =======================
        # Train Discriminator
        # =======================
        outputs = D(real_imgs)
        loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_imgs = G(z)
        outputs = D(fake_imgs.detach())
        loss_fake = criterion(outputs, fake_labels)

        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # =======================
        # Train Generator
        # =======================
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_imgs = G(z)
        outputs = D(fake_imgs)

        loss_G = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}]  Loss D: {loss_D.item():.4f},  Loss G: {loss_G.item():.4f}")

    # ── Save checkpoint + images every 10 epochs ────────────────────────
    if epoch % 20 == 0:
        # Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
        }, ckpt_path)
        print(f"  ✅ Checkpoint saved → {ckpt_path}")

        # Generated image grid
        G.eval()
        with torch.no_grad():
            z_fixed = torch.randn(16, latent_dim, device=device)
            sample_imgs = G(z_fixed).view(-1, 1, 28, 28).cpu()
        G.train()

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(sample_imgs[idx][0], cmap='gray')
            ax.axis('off')
        plt.suptitle(f"Vanilla GAN — Epoch {epoch}")
        plt.tight_layout()
        img_path = os.path.join(image_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(img_path, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  🖼️  Image saved   → {img_path}")

# ── Final visualisation ────────────────────────────────────────────────
G.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, device=device)
    fake_images = G(z).view(-1, 1, 28, 28).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i][0], cmap='gray')
    ax.axis('off')
plt.suptitle("Vanilla GAN — Generated Images after 200 Epochs")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "final.png"), dpi=100, bbox_inches='tight')
plt.show()
print(f"All Vanilla GAN images saved in: {image_dir}")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim  = 100
num_classes = 10
batch_size  = 128       # T4-friendly batch size
epochs      = 200
lr          = 0.0002
checkpoint_dir = "/content/checkpoints/cgan"
image_dir      = "/content/images/cgan"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset    = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img, c], dim=1)
        return self.model(x)


G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]

for epoch in range(1, epochs + 1):
    G.train()
    D.train()
    for real_imgs, labels in dataloader:

        real_imgs = real_imgs.view(-1, 784).to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)
        b         = real_imgs.size(0)

        real = torch.ones(b, 1, device=device)
        fake = torch.zeros(b, 1, device=device)

        # Train D
        out       = D(real_imgs, labels)
        loss_real = criterion(out, real)

        z         = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z, labels)
        out       = D(fake_imgs.detach(), labels)
        loss_fake = criterion(out, fake)

        loss_D = loss_real + loss_fake
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # Train G
        z         = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z, labels)
        out       = D(fake_imgs, labels)
        loss_G    = criterion(out, real)

        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    print(f"Epoch {epoch}/{epochs}  LossD={loss_D.item():.4f}  LossG={loss_G.item():.4f}")

    # ── Save checkpoint + images every 10 epochs ────────────────────────
    if epoch % 20 == 0:
        # Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizer_G_state_dict": opt_G.state_dict(),
            "optimizer_D_state_dict": opt_D.state_dict(),
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
        }, ckpt_path)
        print(f"  ✅ Checkpoint saved → {ckpt_path}")

        # One sample per class
        G.eval()
        with torch.no_grad():
            z_fixed  = torch.randn(10, latent_dim, device=device)
            lbl_fixed = torch.arange(10, device=device)
            sample_imgs = G(z_fixed, lbl_fixed).view(-1, 1, 28, 28).cpu()
        G.train()

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for idx, ax in enumerate(axes):
            ax.imshow(sample_imgs[idx][0], cmap='gray')
            ax.set_title(class_names[idx], fontsize=7)
            ax.axis('off')
        plt.suptitle(f"cGAN — Epoch {epoch}")
        plt.tight_layout()
        img_path = os.path.join(image_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(img_path, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  🖼️  Image saved   → {img_path}")

# ── Final visualisation ────────────────────────────────────────────────
G.eval()
with torch.no_grad():
    z      = torch.randn(10, latent_dim, device=device)
    labels = torch.arange(10, device=device)
    imgs   = G(z, labels).view(-1, 1, 28, 28).cpu()

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i][0], cmap='gray')
    ax.set_title(class_names[i], fontsize=7)
    ax.axis('off')
plt.suptitle("cGAN — One sample per class after 200 Epochs")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "final.png"), dpi=100, bbox_inches='tight')
plt.show()
print(f"All cGAN images saved in: {image_dir}")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
batch_size = 128        # T4-friendly batch size
epochs     = 200
lr         = 0.0002
checkpoint_dir = "/content/checkpoints/dcgan"
image_dir      = "/content/images/dcgan"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset    = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.unsqueeze(2).unsqueeze(3)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(1, epochs + 1):
    G.train()
    D.train()
    for real_imgs, _ in dataloader:

        real_imgs = real_imgs.to(device, non_blocking=True)
        b = real_imgs.size(0)

        real = torch.ones(b, 1, device=device)
        fake = torch.zeros(b, 1, device=device)

        # Train D
        out       = D(real_imgs)
        loss_real = criterion(out, real)

        z         = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z)
        out       = D(fake_imgs.detach())
        loss_fake = criterion(out, fake)

        loss_D = loss_real + loss_fake
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # Train G
        z         = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z)
        out       = D(fake_imgs)
        loss_G    = criterion(out, real)

        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    print(f"Epoch {epoch}/{epochs}  LossD={loss_D.item():.4f}  LossG={loss_G.item():.4f}")

    # ── Save checkpoint + images every 10 epochs ────────────────────────
    if epoch % 20 == 0:
        # Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizer_G_state_dict": opt_G.state_dict(),
            "optimizer_D_state_dict": opt_D.state_dict(),
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
        }, ckpt_path)
        print(f"  ✅ Checkpoint saved → {ckpt_path}")

        # Generated image grid
        G.eval()
        with torch.no_grad():
            z_fixed     = torch.randn(16, latent_dim, device=device)
            sample_imgs = G(z_fixed).cpu()
        G.train()

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(sample_imgs[idx][0], cmap='gray')
            ax.axis('off')
        plt.suptitle(f"DCGAN — Epoch {epoch}")
        plt.tight_layout()
        img_path = os.path.join(image_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(img_path, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  🖼️  Image saved   → {img_path}")

# ── Final visualisation ────────────────────────────────────────────────
G.eval()
with torch.no_grad():
    z         = torch.randn(16, latent_dim, device=device)
    fake_imgs = G(z).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_imgs[i][0], cmap='gray')
    ax.axis('off')
plt.suptitle("DCGAN — Generated Images after 200 Epochs")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "final.png"), dpi=100, bbox_inches='tight')
plt.show()
print(f"All DCGAN images saved in: {image_dir}")

