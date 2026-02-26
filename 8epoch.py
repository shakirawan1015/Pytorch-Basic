import os
os.environ["TORCH_COMPILE_BACKEND"] = "eager"  # Forces simple mode, skips sympy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 1. Define some dummy data 📊
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define model, loss, and optimizer 🛠️
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Set the number of epochs 🔢
num_epochs = 10

# 4. The Training Loop with Epochs 🏃‍♂️
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print status at the end of each epoch 🖨️
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f} ✅")

print("Training Finished! 🎉")