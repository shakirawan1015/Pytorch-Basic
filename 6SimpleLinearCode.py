import torch

# 1. Input Data (e.g., Size of houses)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 2. Parameters (The "Rules" the AI learns)
# Weight (w): How much size affects price
w = torch.tensor(2.0) 
# Bias (b): The base price
b = torch.tensor(1.0)

# 3. Prediction Formula (y = w * x + b)
y_pred = x * w + b

# 4. See the result
print("Predicted Prices:", y_pred)
print("Actual Prices:", x)
print("Loss:", torch.mean((y_pred - x) ** 2))