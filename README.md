# 🎨 FashionMNIST GAN Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ganlab.streamlit.app/)

Welcome to the **FashionMNIST GAN Lab**, an interactive web application built with Streamlit and PyTorch. This project provides a hands-on environment to explore, generate, and evaluate images using four distinct Generative Adversarial Network (GAN) architectures trained from scratch on the FashionMNIST dataset.

### 🌍 Live Demo
You can access the fully deployed, interactive web application right now at:
👉 **[https://ganlab.streamlit.app/](https://ganlab.streamlit.app/)**

## ✨ Features

- **Image Generation:** Instantly generate new clothing items using pre-trained models.
- **Conditional Generation:** Specify exact classes (e.g., Ankle Boot, T-shirt) when using the cGAN model.
- **Discriminator Evaluation:** Upload your own images to test if the trained Discriminator classifies them as "Real" or "Fake".
- **Training Insights:** View interactive charts mapping Generator and Discriminator loss across historical training epochs.
- **Model Checkpoint Explorer:** Traverse history and dynamically reload weights from intermediate epochs to see how the model evolved over time.

---

## 🏗️ Model Architectures

This project explores the progression of GAN architectures, moving from basic multi-layer perceptrons to deep convolutional networks.

### 1. Vanilla GAN
The foundational GAN architecture utilizing fully connected (Linear) layers.
- **Generator Flow:**  
  `Input: Latent Vector(100)` ➔ `Linear(256) ➔ ReLU` ➔ `Linear(512) ➔ ReLU` ➔ `Linear(1024) ➔ ReLU` ➔ `Linear(784) ➔ Tanh` ➔ `Output: Output Image(28x28)`
- **Discriminator Flow:**  
  `Input: Flattened Image(784)` ➔ `Linear(512) ➔ LeakyReLU(0.2)` ➔ `Linear(256) ➔ LeakyReLU(0.2)` ➔ `Linear(1) ➔ Sigmoid` ➔ `Output: Probability (Real/Fake)`

### 2. cGAN (Conditional GAN)
Builds upon the Vanilla GAN by injecting **class labels** into both the Generator and Discriminator, allowing targeted image generation.
- **Generator Flow:**  
  `Input: Latent(100) ⊕ Label Embedding(10)` ➔ `Concat(110)` ➔ `Linear(256) ➔ ReLU` ➔ `Linear(512) ➔ ReLU` ➔ `Linear(784) ➔ Tanh` ➔ `Output: Image (28x28)`
- **Discriminator Flow:**  
  `Input: Image(784) ⊕ Label Embedding(10)` ➔ `Concat(794)` ➔ `Linear(512) ➔ LeakyReLU(0.2)` ➔ `Linear(256) ➔ LeakyReLU(0.2)` ➔ `Linear(1) ➔ Sigmoid` ➔ `Output: Probability (Real/Fake)`

### 3. DCGAN (Deep Convolutional GAN)
The most commonly adopted advanced model, utilizing transposed convolutions for significant quality improvements.
- **Generator Flow:**  
  `Input: Latent(100x1x1)` ➔ `ConvTranspose2d(256, 7x7) ➔ BatchNorm2d ➔ ReLU` ➔ `ConvTranspose2d(128, 14x14) ➔ BatchNorm2d ➔ ReLU` ➔ `ConvTranspose2d(1, 28x28) ➔ Tanh` ➔ `Output: Image(1x28x28)`
- **Discriminator Flow:**  
  `Input: Image(1x28x28)` ➔ `Conv2d(128, 14x14) ➔ LeakyReLU` ➔ `Conv2d(256, 7x7) ➔ BatchNorm2d ➔ LeakyReLU` ➔ `Flatten(12544)` ➔ `Linear(1) ➔ Sigmoid` ➔ `Output: Probability (Real/Fake)`

### 4. StyleGAN
A state-of-the-art architecture adapted specifically for FashionMNIST, demonstrating high-quality synthesis through style incorporation and a Wasserstein loss paradigm.
- **Generator Flow:**  
  `Input: Latent(128)` ➔ `Mapping Network (8 layers)` ➔ `AdaIN Injection` ➔ `StyleBlocks (Upsampling 4x4 ➔ 8x8 ➔ 16x16 ➔ 32x32)` ➔ `Center-Crop(28x28)` ➔ `Conv2d(1x1) ➔ Tanh` ➔ `Output: Image(1x28x28)`
- **Discriminator Flow:**  
  `Input: Image(1x28x28)` ➔ `Conv2d(64, 14x14) ➔ LeakyReLU` ➔ `Conv2d(128, 7x7) ➔ InstanceNorm2d ➔ LeakyReLU` ➔ `Conv2d(256, 3x3) ➔ InstanceNorm2d ➔ LeakyReLU` ➔ `Conv2d(512, 3x3) ➔ InstanceNorm2d ➔ LeakyReLU` ➔ `Flatten(4608)` ➔ `Linear(1)` ➔ `Output: Logit Score (Wasserstein)`

---

## 🚀 Running Locally

If you'd like to run this application on your local machine, follow these steps:

### Prerequisites
Make sure you have **Python 3.8+** installed on your system. 

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/FashionMNIST-GAN-Lab.git
cd FashionMNIST-GAN-Lab
```

### 2. Install Dependencies
It's highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Due to how Windows handles DLLs with PyTorch and Python 3.11+, Streamlit must be launched using a wrapper script to import PyTorch correctly before the Streamlit subprocess starts. 

**Do NOT run `streamlit run app.py`**. Instead, use the provided runner:

```bash
python run_app.py
```
> The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Repository Structure

- `app.py`: The main Streamlit web application interface.
- `run_app.py`: The wrapper script resolving PyTorch loading orders on Windows.
- `requirements.txt`: Python dependencies required for the environment.
- `vanilla_gan/`, `cgan/`, `dcgan/`, `stylegan/`: Directories storing the architectures' respective `.pt` model weights and intermediate output samples.

## 📄 License
This project is open-source and available under the terms of the MIT License.
