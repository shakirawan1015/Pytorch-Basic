import torch

x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12]
])

# Guess the output before running:
print(x[1, 3])      
print(x[:, 0])    # to get the first column
print(x[0, :])    # to get the first row
print(x[0:2, 2:4]) 
