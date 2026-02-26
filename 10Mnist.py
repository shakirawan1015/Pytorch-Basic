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

# Model definition
class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()  # Fixed spacing
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # ← Raw logits (NO LogSoftmax)
        )
    
    def forward(self, x):
        return self.net(x)

model = DigitModel().to(device)

# Loss & Optimizer
loss_fn = nn.CrossEntropyLoss()  # ✓ Matches raw logits
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 10  # Fixed variable name
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_Loader):
        data = data.view(data.size(0), -1).to(device)  # Flatten to [batch, 784]
        target = target.to(device)

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
        data = data.view(data.size(0), -1).to(device)
        target = target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(model.state_dict(), "MNIST_Model.pth")
print("💾 Model saved as 'MNIST_Model.pth'")

# Visualize prediction
index = 0
image, true_label = test_data[index]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True Label: {true_label}")
plt.axis("off")
plt.show()

# Predict single image
image_tensor = image.view(1, -1).to(device)  # ✓ Flatten to [1, 784]
with torch.no_grad():
    output = model(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"🔍 Index: {index}")
print(f"Predicted Label: {predicted_label}")
print(f"True Label: {true_label}")