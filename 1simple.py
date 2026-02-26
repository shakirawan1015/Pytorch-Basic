import torch

x = torch.tensor([1,2,3,4,5])
print(x) # to print the tensor
print(x.dtype) # to print the data type of the tensor
print(x.shape) # to print the shape of the tensor
print(x.ndim) # to print the number of dimensions of the tensor
print(x.size()) # to print the size of the tensor
print(x.device) # to print the device on which the tensor is stored
print(x.requires_grad) # to check if the tensor requires gradient
print(x.is_cuda) # to check if the tensor is stored on a CUDA device
print(x.is_sparse) # to check if the tensor is a sparse tensor
print(x.is_quantized) # to check if the tensor is a quantized tensor
print(x.is_complex()) # to check if the tensor is a complex tensor
print(x.is_floating_point()) # to check if the tensor is a floating point tensor
print(x.is_signed()) # to check if the tensor is a signed tensor
   