import torch    
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# Data transforms
transformer = transforms.ToTensor()

train_data = MNIST(root="data", train=True, transform=transformer, download=True)
test_data = MNIST(root="data", train=False, transform=transformer, download=True)

train_Loader = DataLoader(dataset=train_data, batch_size=30, shuffle=True)
test_Loader = DataLoader(dataset=test_data, batch_size=30, shuffle=False)

#  Model definition (CNN Version)
class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers: extract spatial features from images
        self.conv = nn.Sequential(
            # Conv1: 1 channel → 32 filters, 3×3 kernel, padding to keep 28×28 size
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pool1: reduce 28×28 → 14×14
            nn.MaxPool2d(2),
            
            # Conv2: 32 → 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pool2: reduce 14×14 → 7×7
            nn.MaxPool2d(2),
        )
        # Fully connected layers: classify extracted features
        self.fc = nn.Sequential(
            # 64 filters × 7×7 spatial = 3136 inputs
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # Output: 10 classes (digits 0-9)
        )
    
    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        x = self.conv(x)           # → [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)  # Flatten → [batch, 3136]
        return self.fc(x)          # → [batch, 10]

model = DigitModel().to(device)

# Loss & Optimizer
loss_fn = nn.CrossEntropyLoss()  # ✓ Matches raw logits
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_Loader):
        # ✅ CNN: NO flattening here! Keep [batch, 1, 28, 28] for conv layers
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_Loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_Loader:
        # ✅ Same: keep 4D shape for CNN
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(model.state_dict(), "MNIST_CNN_Model.pth")
print("💾 Model saved as 'MNIST_CNN_Model.pth'")

# Visualize prediction
index = 0
image, true_label = test_data[index]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True Label: {true_label}")
plt.axis("off")
plt.show()

# Predict single image
# ✅ CNN: reshape to [1, 1, 28, 28] = [batch, channel, height, width]
image_tensor = image.view(1, 1, 28, 28).to(device)
with torch.no_grad():
    output = model(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"🔍 Index: {index}")
print(f"Predicted Label: {predicted_label}")
print(f"True Label: {true_label}")