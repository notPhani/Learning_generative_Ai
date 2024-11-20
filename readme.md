# ✨ Handwritten Digit Generator with GANs ✨

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

---

## 🌀 Overview

This project implements a **Generative Adversarial Network (GAN)** that generates realistic images of handwritten digits using the **MNIST** dataset. The GAN architecture consists of two neural networks:

- **Generator**: Creates images resembling handwritten digits.
- **Discriminator**: Distinguishes between real and generated images.

Together, they compete in a game of adversarial training to improve the quality of generated images.

---

## 🎯 Features

- Trained on the popular **MNIST** dataset of handwritten digits (0-9).
- **Generator** produces grayscale images of size `28x28`.
- **Discriminator** achieves an equilibrium by distinguishing real from fake digits.
- Trained for `X epochs` to achieve stable results.

---

## 📊 Architecture

### Generator

The **Generator** uses transposed convolution layers to upsample noise vectors (`z`) into digit-like images.

Key layers:
- Dense layer with a latent space input (`z-dim = 100`).
- Batch Normalization for stability.
- ReLU activation to model non-linearity.
- Transposed Convolution layers for upsampling.

### Discriminator

The **Discriminator** uses convolutional layers to downsample and classify images as real or fake.

Key layers:
- Conv2D layers with LeakyReLU activation for feature extraction.
- Dropout layers to prevent overfitting.
- A Sigmoid output neuron for binary classification.

---

## 🔧 How It Works

1. **Training Loop**:
    - Generator creates fake images from random noise.
    - Discriminator evaluates real vs. fake images.
    - Loss is calculated for both networks, and weights are updated.
2. **Objective**:
    - Generator minimizes the Discriminator's ability to distinguish fake images.
    - Discriminator maximizes its ability to classify real vs. fake.
    - 
---

## 🔧 Requirements

Make sure to have the following installed:

- Python >= 3.7
- TensorFlow >= 2.4.0
- NumPy
- Matplotlib

---

## I have learnt something and so can you!! Happy Coding

Install dependencies and run the file using:


```bash
pip install -r requirements.txt
python3 main.py





