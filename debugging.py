import torch

def compare_1d_tensors(tensor1, tensor2, precision=4):
    """
    Prints two 1D tensors horizontally, one above the other,
    for easy visual comparison.
    """
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("Only 1D tensors are supported.")
    
    if tensor1.size(0) != tensor2.size(0):
        raise ValueError("Tensors must be the same length for comparison.")

    fmt = f"{{:>{precision + 3}.{precision}f}}"
    
    str1 = " ".join(fmt.format(val.item()) for val in tensor1)
    str2 = " ".join(fmt.format(val.item()) for val in tensor2)

    print("Tensor 1:")
    print(str1)
    print("Tensor 2:")
    print(str2)