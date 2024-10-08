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
    xps15
    10 -- 23ms ???
    50 -- 23ms
    99 -- 20ms
    100 -- 10ms (??)

    rpi4
    sparsity -- ms
    10      27
    30      22
    50      13
    70      7
    99      2
    '''
    # for sparsity in [50, 70, 95, 99]:   # cannot do this; will cache prevoius loaded pages
    # for sparsity in [10]:
    for sparsity in [30]:
    # for sparsity in [50]:
    # for sparsity in [70]:
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
        print(f"tensor creation time: {execution_time*1000:.2f} ms")

        # breakpoint()
        indices=gen_index(4*D, 100-sparsity)
        # print(indices)
        # breakpoint()

        start_time = time.perf_counter()
        # extracted_columns = mapped_tensor[:, indices]
        extracted_columns = mapped_tensor[indices, :]   # rows
        extracted_columns = extracted_columns + extracted_columns
        end_time = time.perf_counter()

        print("extracted:", extracted_columns.shape) 
        execution_time = end_time - start_time
        print(f"sparsity {sparsity}, time: {execution_time*1000:.2f} ms")

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

'''
rpi4
(1k,1k) x (1k)      4ms
(4k,1k) x (1k)      9ms
(4k,1k) x (4k,1k)^T 720ms (slow
'''
def test_matmul(dtype=torch.float32):
    shape1=(D,4*D)
    shape2=(D,D)        # 75% sparse

    ffn = torch.randn(shape1, dtype=dtype)
    ffn1 = torch.randn(shape1, dtype=dtype).t()
    ffn2 = torch.randn(shape2, dtype=dtype)         # emulate sparse 

    shape2=(1,D)
    input = torch.randn(shape2, dtype=dtype)
    # input = torch.zeros(shape2, dtype=dtype)

    start_time = time.perf_counter()
    # yyy =ffn@ffn1     # test: matmat
    xxx = input@ffn     # test: matvec, dense
    # xxx = input@ffn2     # test: matvec, sparse
    end_time = time.perf_counter()
    # print(f"xxx {xxx[0]}")
    # breakpoint()
    # print(extracted_columns.shape) 
    execution_time = end_time - start_time
    print(f"matvec compute time: {execution_time*1000:.2f} ms")

'''
rpi4
(4k,1k) x (1k)      8ms     # no much diff from in-mem matmul?? async paging under the hood?
'''
# dense, mmap and matvec
#  4k,1k mat loaded from disk, mmap'd
def load_and_matmul(file_path, tensor_shape, dtype=torch.float32):
    
    start_time = time.perf_counter()
    mapped_tensor = torch.from_file(file_path, dtype=dtype, size=tensor_shape[0] * tensor_shape[1])
    # Reshape to the original tensor shape
    ffn = mapped_tensor.view(tensor_shape)   # does this touch the tensor? 
    start_time1 = time.perf_counter()

    shape2=(1,D)  # vector 
    input = torch.randn(shape2, dtype=dtype)
    # input = torch.zeros(shape2, dtype=dtype)

    # start_time = time.perf_counter()
    xxx = input@ffn     # test: matvec, dense
    xxx += xxx # make sure we do the computation
    end_time = time.perf_counter()

    t1 = start_time1 - start_time
    t2 = end_time - start_time1
    print(f"creation: {t1*1000:.2f} ms, matmul (pg fault): {t2*1000:.2f} ms")    

if __name__ == "__main__":
    # create_random_tensor_and_save(file_path, tensor_shape, dtype)
    # load_mmap_tensor(file_path, tensor_shape)
    test_matmul()
    # load_and_matmul(file_path, (D,4*D), dtype)