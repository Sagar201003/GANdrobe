import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zipfile
import io
import os
import glob
import re

# ==========================================
# Application Configurations & Settings
# ==========================================
st.set_page_config(page_title="GAN Lab", page_icon="🎨", layout="wide")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and Constants
BASE_DIR = r"c:\Users\shiva\OneDrive\Desktop\GAN_Project"
MODEL_DIRS = {
    "Vanilla GAN": os.path.join(BASE_DIR, "vanilla_gan"),
    "cGAN": os.path.join(BASE_DIR, "cgan"),
    "DCGAN": os.path.join(BASE_DIR, "dcgan")
}
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Transform for Discriminator Input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# Model Architectures
# ==========================================

# --- Vanilla GAN ---
class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
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

class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
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

# --- cGAN ---
class CGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
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

class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
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

# --- DCGAN ---
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
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

class DCGANDiscriminator(nn.Module):
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

# ==========================================
# Helper Functions
# ==========================================

@st.cache_resource(show_spinner="Loading Model...")
def load_model(model_name, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None, None
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if model_name == "Vanilla GAN":
        G = VanillaGenerator().to(device)
        D = VanillaDiscriminator().to(device)
    elif model_name == "cGAN":
        G = CGANGenerator().to(device)
        D = CGANDiscriminator().to(device)
    elif model_name == "DCGAN":
        G = DCGANGenerator().to(device)
        D = DCGANDiscriminator().to(device)
        
    G.load_state_dict(ckpt["generator_state_dict"])
    D.load_state_dict(ckpt["discriminator_state_dict"])
    
    G.eval()
    D.eval()
    
    return G, D, ckpt

def get_available_epochs(model_name):
    weights_dir = os.path.join(MODEL_DIRS[model_name], "weights")
    if not os.path.exists(weights_dir):
        return []
    files = glob.glob(os.path.join(weights_dir, "epoch_*.pt"))
    epochs = []
    for f in files:
        match = re.search(r"epoch_(\d+).pt", f)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)

def denormalize(tensor):
    return (tensor + 1) / 2.0

# ==========================================
# Main App Structure
# ==========================================

st.title("GAN Lab — FashionMNIST")
st.markdown("Interactive interface for Vanilla GAN, cGAN, and DCGAN trained on the FashionMNIST dataset.")

# --- Sidebar ---
st.sidebar.title("Controls")
st.sidebar.markdown(f"**Active Device:** `{'CUDA' if device.type == 'cuda' else 'CPU'}`")

# Model Selection
selected_model = st.sidebar.selectbox("Select Model", ["Vanilla GAN", "cGAN", "DCGAN"])

# Epoch Selection
available_epochs = get_available_epochs(selected_model)
if not available_epochs:
    st.sidebar.warning(f"⚠️ No checkpoints found for {selected_model}. Please ensure .pt files are in the {selected_model}/weights directory.")
    selected_epoch = None
    G, D, current_ckpt = None, None, None
else:
    epoch_options = {f"Epoch {e}": e for e in available_epochs}
    selected_epoch_str = st.sidebar.selectbox("Select Epoch", list(epoch_options.keys()), index=len(epoch_options)-1)
    selected_epoch = epoch_options[selected_epoch_str]
    
    if st.sidebar.button("Load Model", type="primary"):
        # The caching mechanism handles not reloading if args are same
        pass

    ckpt_path = os.path.join(MODEL_DIRS[selected_model], "weights", f"epoch_{selected_epoch:03d}.pt")
    G, D, current_ckpt = load_model(selected_model, ckpt_path)

    if G and D and current_ckpt:
        st.sidebar.success(f"✅ Loaded {selected_model} — Epoch {selected_epoch}")
        
        col1, col2 = st.sidebar.columns(2)
        # Handle cases where older checkpoints might not have loss info
        loss_G_val = current_ckpt.get("loss_G", "N/A")
        loss_D_val = current_ckpt.get("loss_D", "N/A")
        
        if loss_G_val != "N/A": col1.metric("Loss G", f"{loss_G_val:.4f}")
        else: col1.metric("Loss G", "N/A")
            
        if loss_D_val != "N/A": col2.metric("Loss D", f"{loss_D_val:.4f}")
        else: col2.metric("Loss D", "N/A")

st.markdown("---")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🎨 Image Generation", "🔍 Real vs Fake Classification"])

# === Tab 1: Image Generation ===
with tab1:
    st.header("Generate Images")
    
    if not G:
        st.warning("Please load a model first from the sidebar.")
    else:
        # Controls setup based on model
        gen_col1, gen_col2 = st.columns(2)
        
        with gen_col1:
            num_images = st.slider("Number of Images to Generate", min_value=1, max_value=16, value=16)
        
        selected_class_idx = 0
        if selected_model == "cGAN":
            with gen_col2:
                selected_class = st.selectbox("Select Class to Generate", CLASS_NAMES)
                selected_class_idx = CLASS_NAMES.index(selected_class)
                
        if st.button("Generate Images", type="primary", use_container_width=True):
            with torch.no_grad():
                z = torch.randn(num_images, 100, device=device)
                if selected_model == "cGAN":
                    labels = torch.full((num_images,), selected_class_idx, dtype=torch.long, device=device)
                    fake_images = G(z, labels)
                else:
                    fake_images = G(z)
                
                # Reshape if necessary (Vanilla GAN outputs flat 784)
                if selected_model in ["Vanilla GAN", "cGAN"]:
                    fake_images = fake_images.view(-1, 1, 28, 28)
                
                fake_images = fake_images.cpu()
            
            # Display logic
            cols = st.columns(4)
            imgs_for_zip = []
            
            for idx in range(num_images):
                img_tensor = fake_images[idx]
                img_display = denormalize(img_tensor).clamp(0, 1).squeeze().numpy()
                
                # Convert to PIL for zip export
                pil_img = Image.fromarray((img_display * 255).astype("uint8"), mode="L")
                imgs_for_zip.append((f"generated_{idx+1}.png", pil_img))
                
                with cols[idx % 4]:
                    st.image(img_display, caption=f"Image {idx+1}")
            
            # Download Zip
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file_name, p_img in imgs_for_zip:
                    img_buffer = io.BytesIO()
                    p_img.save(img_buffer, format="PNG")
                    zip_file.writestr(file_name, img_buffer.getvalue())
            
            st.download_button(
                label="📥 Download All (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"generated_images_{selected_model.replace(' ', '_')}.zip",
                mime="application/zip",
                use_container_width=True
            )

# === Tab 2: Classification ===
with tab2:
    st.header("Discriminator Evaluation: Real or Fake?")
    
    if not D:
        st.warning("Please load a model first from the sidebar.")
    else:
        # For cGAN, we need a class label for the discriminator
        cgan_class_idx = 0
        if selected_model == "cGAN":
            cgan_class = st.selectbox("Assumed Class for Evaluation", CLASS_NAMES)
            cgan_class_idx = CLASS_NAMES.index(cgan_class)
            st.info(f"Evaluating as if the image represents '{cgan_class}'.")

        uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files:
            real_count = 0
            
            cols = st.columns(min(len(uploaded_files), 4))
            
            for i, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                orig_size = img.size
                
                if orig_size != (28, 28):
                    st.warning(f"⚠️ Image '{uploaded_file.name}' resized from {orig_size[0]}x{orig_size[1]} to 28x28")
                
                # Preprocess
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    if selected_model == "Vanilla GAN":
                        flat_input = input_tensor.view(-1, 784)
                        confidence = D(flat_input).item()
                    elif selected_model == "cGAN":
                         flat_input = input_tensor.view(-1, 784)
                         label_tensor = torch.tensor([cgan_class_idx], dtype=torch.long, device=device)
                         confidence = D(flat_input, label_tensor).item()
                    elif selected_model == "DCGAN":
                        confidence = D(input_tensor).item()
                
                is_real = confidence >= 0.5
                if is_real: real_count += 1
                
                with cols[i % 4]:
                    st.image(img.resize((150, 150)), caption=uploaded_file.name)
                    
                    if is_real:
                        st.markdown(f"**🟢 Real — {confidence * 100:.1f}%**")
                        # Streamlit progress needs value between 0.0 and 1.0
                        st.progress(float(confidence))
                    else:
                        st.markdown(f"**🔴 Fake — {(1 - confidence) * 100:.1f}%**")
                        st.progress(float(1.0 - confidence))
            
            # Overall metrics
            st.markdown("---")
            accuracy = (real_count / len(uploaded_files)) * 100
            st.metric("Overall Authenticity (Real)", f"{accuracy:.1f}%")
            st.caption("Accuracy here reflects the Discriminator's confidence — 100% Real means the Discriminator was fully fooled by the fakes, or correctly identified your real images.")

st.markdown("---")

# ==========================================
# Training History Panel
# ==========================================
st.header("Training Progress")

if not available_epochs:
    st.info("No training data available.")
else:
    hist_col1, hist_col2 = st.columns([2, 1])
    
    with hist_col1:
        # Load all history efficiently
        loss_G_history = []
        loss_D_history = []
        epoch_history = []
        
        for e in available_epochs:
            c_path = os.path.join(MODEL_DIRS[selected_model], "weights", f"epoch_{e:03d}.pt")
            if os.path.exists(c_path):
                # Using pure torch.load just reading metadata is faster, but we'll do standard
                ckpt = torch.load(c_path, map_location="cpu", weights_only=True)
                if "loss_G" in ckpt and "loss_D" in ckpt:
                    epoch_history.append(e)
                    loss_G_history.append(ckpt["loss_G"])
                    loss_D_history.append(ckpt["loss_D"])
        
        if epoch_history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epoch_history, y=loss_G_history, mode='lines', name='Generator Loss'))
            fig.add_trace(go.Scatter(x=epoch_history, y=loss_D_history, mode='lines', name='Discriminator Loss'))
            fig.update_layout(title=f"Training Loss — {selected_model}", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)
            
            metric_cols = st.columns(3)
            metric_cols[0].metric("Total Epochs Trained", max(epoch_history))
            metric_cols[1].metric("Final Loss G", f"{loss_G_history[-1]:.4f}")
            metric_cols[2].metric("Final Loss D", f"{loss_D_history[-1]:.4f}")
        else:
            st.info("Loss history not present in available checkpoints.")
            
    with hist_col2:
        if selected_epoch is not None:
            # Try to find target image
            img_path = os.path.join(MODEL_DIRS[selected_model], "results", f"epoch_{selected_epoch:03d}.png")
            if os.path.exists(img_path):
                sample_img = Image.open(img_path)
                st.image(sample_img, caption=f"Training Sample — Epoch {selected_epoch}", use_column_width=True)
            else:
                st.warning(f"No sample image found at `results/epoch_{selected_epoch:03d}.png`.")
