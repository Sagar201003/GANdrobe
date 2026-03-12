# 🎨 FashionMNIST GAN Lab

Welcome to the **FashionMNIST GAN Lab**, an interactive web application built with Streamlit and PyTorch. This project provides a hands-on environment to explore, generate, and evaluate images using three distinct Generative Adversarial Network (GAN) architectures trained from scratch on the FashionMNIST dataset.

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
- **Generator:** Uses a 100-dimensional latent vector (`z`) expanding through hidden layers (`256 -> 512 -> 1024 -> 784`) with LeakyReLU activations, culminating in a `Tanh` layer to produce a flat $28 \times 28$ image.
- **Discriminator:** A simple Multi-Layer Perceptron (MLP) taking a flattened 784-pixel input, reducing it through dense layers (`512 -> 256 -> 1`) using a `Sigmoid` function to output a probability (Real vs. Fake).

### 2. cGAN (Conditional GAN)
Builds upon the Vanilla GAN by injecting **class labels** into both the Generator and Discriminator, allowing targeted image generation.
- **Generator:** Concatenates the 100-dimensional latent vector with a learned 10-dimensional class embedding. This combined input is passed through dense layers to dictate *what* specific FashionMNIST item to generate.
- **Discriminator:** Receives both the generated/real image and its corresponding class label embedding. It learns to determine if the image is not only real, but also if it correctly matches the given conditional label.

### 3. DCGAN (Deep Convolutional GAN)
The most advanced model in this repository, utilizing transposed convolutions for significant quality improvements.
- **Generator:** Removes fully connected layers in favor of `ConvTranspose2d`. It takes the $100 \times 1 \times 1$ latent space and upsamples it using fractional-strided convolutions (`7x7 -> 14x14 -> 28x28`), utilizing Batch Normalization and ReLU.
- **Discriminator:** Acts as a standard CNN classifier using strided `Conv2d` layers downsampling the image (`28x28 -> 14x14 -> 7x7`), utilizing LeakyReLU and Batch Normalization, culminating in a flattened state mapped to a single Sigmoid output.

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
- `vanilla_gan/`, `cgan/`, `dcgan/`: Directories storing the architectures' respective `.pt` model weights and intermediate output samples.

## 📄 License
This project is open-source and available under the terms of the MIT License.
