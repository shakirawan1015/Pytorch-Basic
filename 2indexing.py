import torch

x = torch.tensor([1,2,3,4,5])
print("Element at index 1:", x[1])
print("First 3 elements:", x[0:3])
print("Last element:", x[-1])
print("Every 2nd element:", x[::2])