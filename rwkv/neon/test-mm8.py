import torch
import mm_fp16i8
import inspect
import time

# Print all attributes in module mm_fp16i8
attributes = inspect.getmembers(mm_fp16i8)
for attribute in attributes:
    print(attribute[0])

# below: basically x @ w, x-input, w-weights
#   ry,rx: scaling factors; 
#       rx: liekly applied to input matrix x
#       ry: likely applied to weight matrix w
#   my,mx: biases. (cf above)
#   it's common to scale input & weights separately 
# ex shape: w shape (768,64k) and mx shape (64k) ry shape (768)

def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# Example data
N = 1024
M = 512
# x_fp16 = torch.randn(N, dtype=torch.float16)
# w_uint8 = torch.randint(0, 256, (N, M), dtype=torch.uint8)
# mx_fp16 = torch.randn(M, dtype=torch.float16)
# rx_fp16 = torch.randn(M, dtype=torch.float16)
# my_fp16 = torch.randn(N, dtype=torch.float16)
# ry_fp16 = torch.randn(N, dtype=torch.float16)

# --- below works 
'''
# x_fp16 = torch.randn(N, M, dtype=torch.float16)
x_fp16 = torch.randn(M, dtype=torch.float16)
w_uint8 = torch.randint(0, 256, (M, N), dtype=torch.uint8)
mx_fp16 = torch.randn(N, dtype=torch.float16)
rx_fp16 = torch.randn(N, dtype=torch.float16)
my_fp16 = torch.randn(N, dtype=torch.float16)
ry_fp16 = torch.randn(N, dtype=torch.float16)
'''

############################################################
# mini test case
N = 2
M = 2

# Input vector x_fp16 of size N
x_fp16 = torch.tensor([1.0, 2.0], dtype=torch.float16)

# Weight matrix w_uint8 of size N x M
w_uint8 = torch.tensor([[10, 20], [30, 40]], dtype=torch.uint8)

# mx_fp16 and my_fp16 of size M and N respectively
mx_fp16 = torch.tensor([0.1, 0.2], dtype=torch.float16)  # size M
rx_fp16 = torch.tensor([0.01, 0.02], dtype=torch.float16)  # size M

my_fp16 = torch.tensor([0.001, 0.002], dtype=torch.float16)  # size N
ry_fp16 = torch.tensor([0.0001, 0.0002], dtype=torch.float16)  # size N

# Expected output yy: tensor([0.3050, 0.6050], dtype=torch.float16)
'''
(Pdb) yy
tensor([0.3030, 0.6064], dtype=torch.float16)
(Pdb) y
tensor([0.3052, 0.6055], dtype=torch.float16)
'''

############################################################
N = 10
M = 20

# b (N,M) (786,768*3.5) mx (M) rx (M) my (N,1) ry (N,1)

# Input vector x_fp16 of size N
x_fp16 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float16)

# Weight matrix w_uint8 of size N x M
w_uint8 = torch.arange(1, N * M + 1, dtype=torch.uint8).reshape(N, M)

# mx_fp16 and rx_fp16 of size M
mx_fp16 = torch.linspace(0.1, 0.2, M, dtype=torch.float16)
rx_fp16 = torch.linspace(0.01, 0.02, M, dtype=torch.float16)

# my_fp16 and ry_fp16 of size Nx1
my_fp16 = torch.linspace(0.001, 0.002, N, dtype=torch.float16).unsqueeze(1)
ry_fp16 = torch.linspace(0.0001, 0.0002, N, dtype=torch.float16).unsqueeze(1)

'''
expected:
tensor([ 5.6016,  5.8906,  6.1797,  6.4688,  6.7617,  7.0508,  7.3438,  7.6367,
         7.9258,  8.2109,  8.5000,  8.7969,  9.0859,  9.3750,  9.6641,  9.9609,
        10.2500, 10.5391, 10.8281, 11.1172], dtype=torch.float16)
'''

############################################################

# Measure execution time for torch_mm8_one
start_time = time.time()
yy = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
end_time = time.time()
print(f"Execution time for torch_mm8_one: {end_time - start_time} seconds")

# Measure execution time for mm_fp16i8.mm_one_fp16i8
start_time = time.time()
y = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
end_time = time.time()
print(f"Execution time for mm_fp16i8.mm_one_fp16i8: {end_time - start_time} seconds")


# Compare if yy and y are close enough
if torch.allclose(yy, y, atol=1e-1):
    print("The results are close enough.")
else:
    print("The results are not close enough.")