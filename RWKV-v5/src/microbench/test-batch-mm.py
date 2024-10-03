import torch
import time

'''
purpose: to see if we have any speedup by using batched matmul, to recover from decomposed att/ffn weights...
conclusion: minor speedup, but not much (esp when D is larger) 
'''

# Define the dimensions       M is actually D (feature dim)
# M, N = 1024, 1024//8
M, N = 2048, 2048//8

# Generate random tensors
tensors_a = [torch.randn(M, N) for _ in range(5)]
tensors_b = [torch.randn(N, M) for _ in range(5)]

# Method 1: Iterate the matmul
start_time = time.time()
results_iterate = [torch.matmul(a, b) for a, b in zip(tensors_a, tensors_b)]
time_iterate = time.time() - start_time

# Method 2: Batched matmul
batched_a = torch.stack(tensors_a)
batched_b = torch.stack(tensors_b)

start_time = time.time()
results_batched = torch.bmm(batched_a, batched_b)
time_batched = time.time() - start_time

# Print the results
print(f"M: {M}, N: {N}")
print(f"Time taken for iterative matmul: {time_iterate:.6f} seconds")
print(f"Time taken for batched matmul: {time_batched:.6f} seconds")


'''
results: odroid N2 (cpu)

M: 1024, N: 128     (simlar) 
Time taken for iterative matmul: 0.063073 seconds
Time taken for batched matmul: 0.050011 seconds

M: 2048, N: 256
Time taken for iterative matmul: 0.304577 seconds
Time taken for batched matmul: 0.306522 seconds
'''