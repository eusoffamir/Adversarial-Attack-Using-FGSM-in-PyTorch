# Lab 1: Adversarial Attack Using FGSM in PyTorch

This repository contains the implementation of **Lab 1** for **RBB4633: Next Generation Computing Security (May 2026)**. The objective of this project is to implement and analyze a white-box adversarial attack using the **Fast Gradient Sign Method (FGSM)** on a trained neural network using the MNIST dataset.

## Course & Author Information
* **Course Code:** RBB4633 Next Generation Computing Security
* **Date of Experiment:** 4th June 2026
* **Student Name:** Muhammad Eusoff Amir bin Hazri
* **Student ID:** 22010491
* **Major:** Computer Engineering

---

## Project Overview

Adversarial attacks manipulate input data by adding tiny, strategic perturbations (noise) to standard testing images to trick deep learning models into making completely incorrect predictions. 

This repository executes a white-box **Fast Gradient Sign Method (FGSM)** attack. FGSM utilizes the gradients of the model's loss function with respect to the input image to calculate the exact direction that maximizes classification error.

### The FGSM Equation:
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_{x}J(\theta, x, y))$$

* $x$: Original clean input image
* $\epsilon$: Perturbation magnitude (noise strength)
* $J$: Loss function of the model
* $\theta$: Trained model parameters
* $x_{adv}$: Generated adversarial example

---

## File Structure

```text
├── data/                             # Directory where the MNIST dataset is downloaded
├── simple_nn_mnist.pth               # Pre-trained model weights for SimpleNN
├── Adversarial_06_FGSM_Attack.py     # Main Python script executing the attack
└── README.md                         # Project documentation
