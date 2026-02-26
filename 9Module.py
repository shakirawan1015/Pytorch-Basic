import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility (optional but recommended)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Create model
model = SimpleModel().to(device)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32, device=device)
y = torch.tensor([[150.0], [250.0], [350.0], [450.0], [550.0]], dtype=torch.float32, device=device)

print("🚀 Starting Training...")

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print(f"\n✅ Final Loss: {loss.item():.4f}")

# Test prediction
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[6.0]], dtype=torch.float32, device=device)
    prediction = model(test_input)
    print(f"Prediction for input 6.0: {prediction.item():.2f} (Expected ~650)")

# Show learned parameters
print(f"\n📊 Learned Parameters:")
print(f"Weight: {model.linear.weight.item():.2f} (Expected ~100)")
print(f"Bias: {model.linear.bias.item():.2f} (Expected ~50)")