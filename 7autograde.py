import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([150.0, 250.0, 350.0, 450.0, 550.0])

w = torch.tensor(10.0, requires_grad=True)

learning_rate = 0.01
num_iterations = 10

for i in range(num_iterations):
    # Forward pass: compute predicted y
    y_pred = w * x
    
    # Compute loss (mean squared error)
    loss = ((y_pred - y) ** 2).mean()
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update weights (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # Zero the gradients
    w.grad.zero_()
    
    if i % 1 == 0:
        print(f'Iteration {i}: Loss = {loss.item():.4f}, Weight = {w.item():.4f}')