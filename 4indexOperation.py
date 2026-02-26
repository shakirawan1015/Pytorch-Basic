import torch

a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([10, 20, 30, 40, 50])

#  Addition
print("Add:", a + b)        # tensor([11, 22, 33, 44, 55])

#  Multiplication
print("Multiply:", a * b)   # tensor([10, 40, 90, 160, 250])

#  Subtraction
print("Subtract:", b - a)   # tensor([9, 18, 27, 36, 45])