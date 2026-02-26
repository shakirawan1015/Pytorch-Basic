import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

# Load test data
test_data = MNIST(root="data", train=False, transform=transforms.ToTensor(), download=True)

# Model (must match training)
class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.net(x)

# Load model
device = torch.device('cpu')
model = DigitModel().to(device)
model.load_state_dict(torch.load("MNIST_Model.pth", map_location=device, weights_only=True))
model.eval()

print("✅ Model loaded! Enter index to test (or -1 to quit)\n")

# Test loop
while True:
    index = int(input("Enter image index (0-9999): "))
    
    if index == -1:
        print("👋 Goodbye!")
        break
    
    # Get image
    image, true_label = test_data[index]
    
    # Show image
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"True: {true_label}")
    plt.axis("off")
    plt.show()
    
    # 🔑 Predict (FIXED: inside loop + proper indentation)
    image_tensor = image.view(1, -1).to(device)  # Flatten to [1, 784]
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)  # Convert to probabilities
        predicted = torch.argmax(output, dim=1).item()
        confidence = probs[0][predicted].item() * 100  # Get confidence %
    
    # Result with confidence
    status = "✅ Correct" if predicted == true_label else "❌ Wrong"
    print(f"Predicted: {predicted} | True: {true_label} | Confidence: {confidence:.1f}% | {status}\n")