'''
load only certain columns of a tensor file (mmap)
and measure overhead etc
'''

import sys
import torch
import time 
import random
import mmap
import os
import torch.autograd.profiler as profiler

# Parameters of the tensor
D=1024
dtype = torch.float16  # Data type of the tensor elements
tensor_shape = (4*D, D)  # Shape of the 2D tensor (rows, columns)
file_path = '/tmp/large_tensor_file.bin'  # Path to the tensor file

# D: 0,D -- range
# k: %k numbers in 0,D
# return k random numbers in 0,D, sorted
def gen_index(D, k): 
    p = int((k / 100) * D)
    # Generate p unique random numbers in the range (0, D)
    random_numbers = random.sample(range(D), p)
    
    # Sort the list in ascending order
    sorted_numbers = sorted(random_numbers)
    
    return sorted_numbers    

'''
mmap a tensor file and extract certain columns (ratio controlled by sparsity)
'''
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
    # for sparsity in [50, 70, 95, 99]:   # cannot do this
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
        mapped_tensor = mapped_tensor.view(tensor_shape)   # does this touch the tensor?  -- seems no
        end_time = time.perf_counter()
        # print(extracted_columns.shape) 
        execution_time = end_time - start_time
        print(f"tensor creation time: {execution_time*1000:.2f} ms")

        # breakpoint()
        indices=gen_index(4*D, 100-sparsity)
        # print(indices)
        # breakpoint()

        # Extract only the specified columns from the tensor (ratio of sparsity)

        start_time = time.perf_counter()
        # extracted_columns = mapped_tensor[:, indices]
        extracted_columns = mapped_tensor[indices, :]   # rows
        extracted_columns = extracted_columns + extracted_columns
        end_time = time.perf_counter()

        print("extracted:", extracted_columns.shape) 
        execution_time = end_time - start_time
        print(f"sparsity {sparsity}, time: {execution_time*1000:.2f} ms")

# mmap a tensor file, and do matmul with a vector
def load_mmap_tensor_matmul(file_path, tensor_shape):
    '''
    xps15

    rpi4
    sparsity -- ms
    '''

    start_time = time.perf_counter()
    mapped_tensor = torch.from_file(file_path, dtype=dtype, size=tensor_shape[0] * tensor_shape[1])
    mapped_tensor = mapped_tensor.view(tensor_shape)   # does this touch the tensor?  -- seems no
    end_time = time.perf_counter()
    # print(extracted_columns.shape) 
    execution_time = end_time - start_time
    print(f"tensor creation time: {execution_time*1000:.2f} ms")


    # --- matvec on the sparse tensor 
    input = torch.randn(tensor_shape[1], dtype=dtype)
    start_time = time.perf_counter()
    xxx = mapped_tensor @ input
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"matvec compute time: {execution_time*1000:.2f} ms")


'''
mmap a tensor file, take the file, create a tensor out of it 
    rpi4
    sparsity -- ms
    30      22
    (no diff from load_mmap_tensor() which creates a tensor from the file)
'''
def load_tensor_with_anonymous_mapping(file_path, tensor_shape):
    for sparsity in [30]:
        # Memory-map the tensor file
        with open(file_path, "r+b") as f:
            # Memory-map the file, size 0 means whole file
            mmapped_file = mmap.mmap(f.fileno(), 0)
            
            # Create a tensor from the memory-mapped region
            start_time = time.perf_counter()
            mapped_tensor = torch.frombuffer(mmapped_file, dtype=dtype, count=tensor_shape[0] * tensor_shape[1])
            mapped_tensor = mapped_tensor.view(tensor_shape)  # Reshape to the original tensor shape
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"tensor creation time: {execution_time*1000:.2f} ms")

            ########################################

            indices = gen_index(tensor_shape[0], 100-sparsity)

            start_time = time.perf_counter()
            extracted_columns = mapped_tensor[indices, :]  # Extract rows            
            end_time = time.perf_counter()

            print("extracted:", extracted_columns.shape)
            execution_time = end_time - start_time
            print(f"sparsity {sparsity}, time: {execution_time*1000:.2f} ms")

            ########################################

            # Create an anonymous mapping that overlaps with the mapped_file
            # NOT WORKING ... READ CANNOT RETURN 0 
            anon_mmap = mmap.mmap(f.fileno(), 4096, access=mmap.ACCESS_READ, offset=0)
            row0 = mapped_tensor[0, :]
            print(f"row0 {row0}")
            # extracted_columns += torch.frombuffer(anon_mmap, dtype=dtype, count=4096 // dtype().element_size())

            # Clean up
            anon_mmap.close()
            mmapped_file.close()

'''
another attempt --- worked

In Python, you can use the mmap module to map memory, but by default, the module
doesn’t allow you to directly specify a memory address or set flags like
MAP_FIXED. However, you can use the ctypes or ctypes.util module to interface
with the mmap system call at a lower level and pass custom flags.
'''
import ctypes

# Constants (these can vary between systems, check your system headers)
PROT_READ = 0x1
PROT_WRITE = 0x2
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20
MAP_FIXED = 0x10

# Get libc
libc = ctypes.CDLL("libc.so.6")

# mmap syscall: void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
def custom_mmap(addr, length, prot, flags, fd=-1, offset=0):
    libc.mmap.restype = ctypes.c_void_p  # return type: void*
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    
    result = libc.mmap(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                       ctypes.c_int(prot), ctypes.c_int(flags),
                       ctypes.c_int(fd), ctypes.c_size_t(offset))
    
    if result == -1:
        raise OSError("mmap failed")
    return result

# same as above, but return <Python-buf, raw-buf>. Python buf can be used 
# as torch.from_buffer() input
# mmap syscall: void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
def custom_mmap_python(addr, length, prot, flags, fd=-1, offset=0):
    libc.mmap.restype = ctypes.c_void_p  # return type: void*
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    
    result = libc.mmap(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                       ctypes.c_int(prot), ctypes.c_int(flags),
                       ctypes.c_int(fd), ctypes.c_size_t(offset))
    
    if result == -1:
        raise OSError("mmap failed")
    
    # Create a ctypes type that matches the buffer length and map to the result address
    ctypes_array_type = (ctypes.c_char * length)
    ctypes_array = ctypes_array_type.from_address(result)
    
    # Return a memoryview, which is a buffer-like object
    return memoryview(ctypes_array), result 

# try mmap single memory region (a row)
# first mmap the tensor file, then mmap anon region
def load_tensor_with_anonymous_mapping2(file_path, tensor_shape):
    for sparsity in [30]:
        # Memory-map the tensor file
        with open(file_path, "r+b") as f:
            # Memory-map the file, size 0 means whole file
            mmapped_file = mmap.mmap(f.fileno(), 0)
            
            # Create a tensor from the memory-mapped region
            start_time = time.perf_counter()
            mapped_tensor = torch.frombuffer(mmapped_file, dtype=dtype, count=tensor_shape[0] * tensor_shape[1])
            mapped_tensor = mapped_tensor.view(tensor_shape)  # Reshape to the original tensor shape
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"tensor creation time: {execution_time*1000:.2f} ms")

            # Get the memory address of the tensor
            tensor_address = mapped_tensor.data_ptr()

            # Define memory mapping parameters
            length = 4096
            prot = PROT_READ
            flags = MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS

            # Map memory at the tensor's address
            mapped_addr = custom_mmap(tensor_address, length, prot, flags)

            print(f"Memory mapped at: {hex(mapped_addr)}")
            print(f"Tensor memory address: {hex(tensor_address)}")

            row0 = mapped_tensor[0, :]
            print(f"row0 {row0}")
            rowN = mapped_tensor[-1, :]
            print(f"rowN {rowN}")
            mmapped_file.close()

# benchmark
# first mmap the tensor file, then mmap anon region
def load_tensor_with_anonymous_mapping_bench(file_path, tensor_shape):
    for sparsity in [50]:
        # Memory-map the tensor file
        with open(file_path, "r+b") as f:
            # Memory-map the file, size 0 means whole file
            mmapped_file = mmap.mmap(f.fileno(), 0)
            
            # Create a tensor from the memory-mapped region
            start_time = time.perf_counter()
            mapped_tensor = torch.frombuffer(mmapped_file, dtype=dtype, count=tensor_shape[0] * tensor_shape[1])
            mapped_tensor = mapped_tensor.view(tensor_shape)  # Reshape to the original tensor shape
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"tensor creation time: {execution_time*1000:.2f} ms")

            row_size_bytes = mapped_tensor.element_size() * tensor_shape[1]

            indices = gen_index(tensor_shape[0], sparsity)

            # Get the memory address of the tensor
            tensor_address = mapped_tensor.data_ptr()
            # Define memory mapping parameters
            length = row_size_bytes
            prot = PROT_READ
            flags = MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS

            #### naive -- call mmap() once for each "zero" row
            start_time = time.perf_counter()
            for i in indices:   
                # Map memory at the tensor's address
                mapped_addr = custom_mmap(tensor_address + row_size_bytes*i, 
                                          length, prot, flags)
            end_time = time.perf_counter()
            print(f"mmapped {len(indices)} rows"
                f"in {1000*(end_time-start_time):.2f} ms"
                f"({1000*(end_time-start_time)/len(indices):.4f} ms per row)")

            # --- sanity check 
            row0 = mapped_tensor[indices[0], :]
            print(f"row0 {row0}")
            rowN = mapped_tensor[-1, :]
            print(f"rowN {rowN}")
            
            # --- matvec on the sparse tensor 
            input = torch.randn(tensor_shape[1], dtype=dtype)
            start_time = time.perf_counter()
            xxx = mapped_tensor @ input
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"matvec compute time: {execution_time*1000:.2f} ms")

            mmapped_file.close()


# benchmark2 --- create a big anon mapping, atop it, mmap the tensor file
# first mmap the big anon region, then mmap regions from the tensor file 
def load_tensor_with_anonymous_mapping_bench2(file_path, tensor_shape):
    for sparsity in [50]:
        # Memory-map the tensor file
        with open(file_path, "r+b") as f:

            # Define memory mapping parameters
            row_size_bytes = torch._utils._element_size(dtype) * tensor_shape[1]
            length = row_size_bytes * tensor_shape[0]

            prot = PROT_READ | PROT_WRITE
            flags = MAP_PRIVATE | MAP_ANONYMOUS
            # Create one large anonymous mmap area
            anon_mmap_py, anon_mmap = custom_mmap_python(-1, length, prot, flags)
            print(f"Memory mapped at: {hex(anon_mmap)}")

            #### Map parts of the file to the anonymous mmap area
            indices = gen_index(tensor_shape[0], 100-sparsity)

            start_time = time.perf_counter()
            for i in indices[:-1]:    # last row is not mapped. for testing 
                # Map memory at the tensor's address
                _, mapped_addr = custom_mmap_python(anon_mmap + row_size_bytes * i, 
                            row_size_bytes, prot, MAP_FIXED | MAP_PRIVATE, f.fileno(), row_size_bytes * i)
            end_time = time.perf_counter()
            print(f"mmapped {len(indices)} rows"
            f"in {1000*(end_time-start_time):.2f} ms"
            f"({1000*(end_time-start_time)/len(indices):.4f} ms per row)")

            # Create a tensor from the anonymous mmap area
            mapped_tensor = torch.frombuffer(anon_mmap_py, dtype=dtype, count=tensor_shape[0] * tensor_shape[1])
            mapped_tensor = mapped_tensor.view(tensor_shape)  # Reshape to the original tensor shape

            # --- sanity check 
            row0 = mapped_tensor[indices[0], :]
            print(f"row0 {row0}")   # should read nonzero
            rowN = mapped_tensor[-1, :]
            print(f"rowN {rowN}")   # should read 0 

            # --- matvec on the sparse tensor 
            input = torch.randn(tensor_shape[1], dtype=dtype)
            start_time = time.perf_counter()
            xxx = mapped_tensor @ input
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"matvec compute time: {execution_time*1000:.2f} ms")            


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
    # Create a tensor with the specified shape, data type, and fill it with a specific value
    specific_value = 1.0  # You can change this value to whatever you need
    random_tensor = torch.full(shape, specific_value, dtype=dtype)
    
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
# create in-mem tensor and matmul, mat-vec
def test_matmul(dtype=torch.float32):
    shape1=(4*D,D)
    shape2=(D,D)        # 75% sparse

    ffn = torch.randn(shape1, dtype=dtype)
    ffn1 = torch.randn(shape1, dtype=dtype).t()
    ffn2 = torch.randn(shape2, dtype=dtype)         # emulate sparse 

    input = torch.randn((D), dtype=dtype)
    # input = torch.zeros(shape2, dtype=dtype)

    start_time = time.perf_counter()
    # yyy =ffn@ffn1     # test: matmat
    xxx = ffn @ input     # test: matvec, dense
    # xxx = input@ffn2     # test: matvec, sparse
    end_time = time.perf_counter()
    # breakpoint()
    # print(f"xxx {xxx[0]}")
    # breakpoint()
    # print(extracted_columns.shape) 
    execution_time = end_time - start_time
    print(f"test_matmul: matvec compute time: {execution_time*1000:.2f} ms")

'''
rpi4
(4k,1k) x (1k)      8ms     # no much diff from in-mem matmul?? async underhood?
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
    if len(sys.argv) > 1 and sys.argv[1] == "--save-tensor":
        create_random_tensor_and_save(file_path, tensor_shape, dtype)
    else:
        test_matmul(dtype=torch.float16)
        test_matmul(dtype=torch.float32)
        # load_and_matmul(file_path, (D,4*D), dtype)
        
        # load_mmap_tensor(file_path, tensor_shape)
        # load_mmap_tensor_matmul(file_path, tensor_shape)

        # load_tensor_with_anonymous_mapping(file_path, tensor_shape)
        # load_tensor_with_anonymous_mapping2(file_path, tensor_shape)
        # load_tensor_with_anonymous_mapping_bench(file_path, tensor_shape)
        # load_tensor_with_anonymous_mapping_bench2(file_path, tensor_shape)