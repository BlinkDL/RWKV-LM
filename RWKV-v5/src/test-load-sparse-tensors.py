'''
load only certain columns of a tensor file (mmap)
and measure overhead etc
'''

import torch
import time 
import random
import torch.autograd.profiler as profiler



# Parameters of the tensor
D=1024
dtype = torch.float16  # Data type of the tensor elements
tensor_shape = (4*D, D)  # Shape of the 2D tensor (rows, columns)
file_path = '/tmp/large_tensor_file.bin'  # Path to the tensor file

# D: 0,D -- range
# k: %k numbers in 0,D
def gen_index(D, k): 
    p = int((k / 100) * D)
    # Generate p unique random numbers in the range (0, D)
    random_numbers = random.sample(range(D), p)
    
    # Sort the list in ascending order
    sorted_numbers = sorted(random_numbers)
    
    return sorted_numbers    

def load_mmap_tensor(file_path, tensor_shape):
    '''
    10 -- 23ms ???
    50 -- 23ms
    99 -- 20ms
    100 -- 10ms (??)
    '''
    # for sparsity in [50, 70, 95, 99]:   # cannot do this
    for sparsity in [10]:
    # for sparsity in [50]:
    # for sparsity in [99]:
    # for sparsity in [100]:
        # Memory-map the tensor from the file
        start_time = time.perf_counter()
        mapped_tensor = torch.from_file(file_path, dtype=dtype, size=tensor_shape[0] * tensor_shape[1])
        # Reshape to the original tensor shape
        mapped_tensor = mapped_tensor.view(tensor_shape)   # does this touch the tensor? 
        end_time = time.perf_counter()
        # print(extracted_columns.shape) 
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        # breakpoint()
        indices=gen_index(4*D, 100-sparsity)
        # print(indices)
        # breakpoint()

        start_time = time.perf_counter()
        # extracted_columns = mapped_tensor[:, indices]
        extracted_columns = mapped_tensor[indices, :]   # rows
        end_time = time.perf_counter()

        print("extracted:", extracted_columns.shape) 
        execution_time = end_time - start_time
        print(f"sparsity {sparsity}, time: {execution_time} seconds")

def create_random_tensor_and_save(file_path, shape, dtype=torch.float32):
    """
    Creates a tensor with random data and writes it to a binary file.
    
    Args:
    - file_path (str): Path to the file where the tensor will be saved.
    - shape (tuple): Shape of the tensor to create.
    - dtype (torch.dtype): Data type of the tensor (default: torch.float32).
    
    Returns:
    - None
    """
    # Create a random tensor with the specified shape and data type
    random_tensor = torch.randn(shape, dtype=dtype)
    
    # Open the file in binary write mode and save the raw tensor data
    with open(file_path, 'wb') as f:
        # Write the raw data (tensor as bytes) to the file
        f.write(random_tensor.numpy().tobytes())

    print(f"Tensor with shape {shape} and dtype {dtype} saved to {file_path}")

def test_matmul(dtype=torch.float32):
    shape1=(D,4*D)
    ffn = torch.randn(shape1, dtype=dtype)
    shape2=(1,D)
    input = torch.randn(shape2, dtype=dtype)

    start_time = time.perf_counter()
    xxx = input@ffn
    end_time = time.perf_counter()
    print(f"xxx {xxx[0]}")
    # breakpoint()
    # print(extracted_columns.shape) 
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    # create_random_tensor_and_save(file_path, tensor_shape, dtype)
    # load_mmap_tensor(file_path, tensor_shape)
    test_matmul()