# ==========================================
# Adversarial_06_FGSM_Attack.py
# Purpose:
# Use trained model + perform FGSM attack
# ==========================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# =========================
# Task 1 – Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Task 2 – Load MNIST Dataset
# =========================
transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True
)


# =========================
# Task 3 – Define SimpleNN
# (REQUIRED when loading model)
# =========================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# Task 4 – Load trained model
# =========================
model = SimpleNN().to(device)

model.load_state_dict(
    torch.load("simple_nn_mnist.pth", map_location=device)
)

model.eval()

print("Model loaded successfully!")


# =========================
# Task 5 – FGSM Attack Function
# =========================
loss_fn = nn.CrossEntropyLoss()

def fgsm_attack(image, epsilon, data_grad):
    sign_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# =========================
# Task 6 – Generate Adversarial Example
# =========================
image, label = next(iter(test_loader))

image, label = image.to(device), label.to(device)

image.requires_grad = True

# Forward pass
output = model(image)
init_pred = output.argmax(dim=1)

# Loss
loss = loss_fn(output, label)

model.zero_grad()
loss.backward()

# Gradient
data_grad = image.grad.data

# FGSM attack
epsilon = 0.2
adv_image = fgsm_attack(image, epsilon, data_grad)

# Re-check prediction
adv_output = model(adv_image)
adv_pred = adv_output.argmax(dim=1)


# =========================
# Task 7 – Results
# =========================
print("True Label:", label.item())
print("Original Prediction:", init_pred.item())
print("Adversarial Prediction:", adv_pred.item())


# =========================
# Task 8 – Visualization
# =========================
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(image.squeeze().detach().cpu().numpy(), cmap='gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(adv_image.squeeze().detach().cpu().numpy(), cmap='gray')
plt.title(f"Adversarial (ε={epsilon})")

plt.show()