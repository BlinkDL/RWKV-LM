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

# x shape can be (batch,D)
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

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
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

yy = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm_fp16i8.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
# breakpoint()

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
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

yy = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm_fp16i8.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
############################################################
N = 50
M = 100

# Input vector x_fp16 of size N
x_fp16 = torch.tensor([i + 1.0 for i in range(N)], dtype=torch.float16)

# Weight matrix w_uint8 of size N x M
w_uint8 = torch.arange(1, N * M + 1, dtype=torch.uint8).reshape(N, M)

# mx_fp16 and rx_fp16 of size M
mx_fp16 = torch.linspace(0.1, 1.5, M, dtype=torch.float16)
rx_fp16 = torch.linspace(0.1, 1.5, M, dtype=torch.float16)

# my_fp16 and ry_fp16 of size Nx1
my_fp16 = torch.linspace(0.1, 1.5, N, dtype=torch.float16).unsqueeze(1)
ry_fp16 = torch.linspace(0.1, 1.5, N, dtype=torch.float16).unsqueeze(1)

'''
expected:
tensor([...], dtype=torch.float16)
'''
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

yy = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm_fp16i8.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
# breakpoint()

############################################################
# neg test case
x_fp16 = torch.tensor([-8.7786e-04, -2.4207e-01, -1.0859e+00,  1.0625e+00, -3.5840e-01,
                       2.9316e+00,  1.3037e+00,  4.5337e-01,  5.3760e-01, -1.4478e-01],
                      dtype=torch.float16)

w_uint8 = torch.tensor([[149, 247, 110,   3,  53, 148,  33, 216,   2,  97],
                        [ 24, 180, 232,  84, 223,  56, 208, 190, 103, 163],
                        [185,  30,  85,  14, 143, 252,  98,  62, 143, 112],
                        [ 19, 185,   4, 176, 115, 195,  32,  87, 132, 202],
                        [ 85,  74, 221, 113, 137,  74, 235, 150,  15, 116],
                        [ 64, 195, 176, 245,  51, 190,  25, 254,  86, 106],
                        [162, 109,  63,  68,  56, 243, 120,  94, 192,   9],
                        [ 80, 121, 178,  22,  23, 148,  35,  56, 153, 103],
                        [ 27,  24, 233,  49,  91,  10, 144, 129,  82, 183],
                        [123,  12, 105, 143, 236, 201, 138, 253, 161,  85]],
                       dtype=torch.uint8)

mx_fp16 = torch.tensor([0.1000, 0.2000, 0.2998, 0.3999, 0.5000, 0.6001, 0.7002, 0.7998, 0.8999,
                        1.0000], dtype=torch.float16)

rx_fp16 = torch.tensor([0.3914, 0.0447, 0.1252, 0.2034, 0.3882, 0.1665, 0.1735, 0.4670, 0.3103,
                        0.3884], dtype=torch.float16)

my_fp16 = torch.tensor([[0.0010],
                        [0.0011],
                        [0.0012],
                        [0.0013],
                        [0.0014],
                        [0.0016],
                        [0.0017],
                        [0.0018],
                        [0.0019],
                        [0.0020]], dtype=torch.float16)

ry_fp16 = torch.tensor([[0.0581],
                        [0.3372],
                        [0.1445],
                        [0.2220],
                        [0.0409],
                        [0.4751],
                        [0.3643],
                        [0.0629],
                        [0.0181],
                        [0.2498]], dtype=torch.float16)

y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

yy = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm_fp16i8.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
print(">here")
# breakpoint()

############################################################
# x (N) w (N,M) (768,768*3.5)  mx (M) rx (M) my (N,1) ry (N,1)
# Example data
N = 1024
# N = 10
# N = 40
M = int(N * 3.5)
# M=10

x_fp16 = torch.randn(N, dtype=torch.float16)
w_uint8 = torch.randint(0, 256, (N, M), dtype=torch.uint8)
# mx_fp16 = torch.randn(M, dtype=torch.float16)
# rx_fp16 = torch.randn(M, dtype=torch.float16)
# my_fp16 = torch.randn((N,1), dtype=torch.float16)
# ry_fp16 = torch.randn((N,1), dtype=torch.float16)

# mx_fp16 and rx_fp16 of size M
mx_fp16 = torch.linspace(0.1, 1.0, M, dtype=torch.float16)
rx_fp16 = torch.rand(M, dtype=torch.float16) * 0.49 + 0.01
# my_fp16 and ry_fp16 of size Nx1
my_fp16 = torch.linspace(0.001, 0.002, N, dtype=torch.float16).unsqueeze(1)
ry_fp16 = torch.rand((N,1), dtype=torch.float16) * 0.49 + 0.01


# y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)

# y1 = torch_mm8_one(
#     x_fp16.to(torch.float), 
#     w_uint8, 
#     mx_fp16.to(torch.float), 
#     rx_fp16.to(torch.float), 
#     my_fp16.to(torch.float), 
#     ry_fp16.to(torch.float)
# )

# yy = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
# yyy = mm_fp16i8.mm_one_fp32i8(
#     x_fp16.to(torch.float), 
#     w_uint8, 
#     mx_fp16.to(torch.float), 
#     rx_fp16.to(torch.float), 
#     my_fp16.to(torch.float), 
#     ry_fp16.to(torch.float)
# )
# print(f"y: {y[:10]}")
# print(f"yy: {yy[:10]}")
# print(f"yyy: {yyy[:10]}")

# breakpoint()

############################################################

# Measure execution time for torch_mm8_one
start_time = time.time()
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
end_time = time.time()
print(f"Execution time for torch_mm8_one: {(end_time - start_time) * 1000:.3f} ms")

# fp16, different tries 
start_time = time.time()
yy1 = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
end_time = time.time()
print(f"Execution time for mm_one_fp16i8    v1: {(end_time - start_time) * 1000:.3f} ms")

start_time = time.time()
yy2 = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 2)
end_time = time.time()
print(f"Execution time for mm_one_fp16i8    v2: {(end_time - start_time) * 1000:.3f} ms")

start_time = time.time()
yy3 = mm_fp16i8.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 3)
end_time = time.time()
print(f"Execution time for mm_one_fp16i8    v3: {(end_time - start_time) * 1000:.3f} ms")

# fp32
start_time = time.time()
yyy = mm_fp16i8.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
end_time = time.time()
print(f"Execution time for mm_one_fp32i8: {(end_time - start_time) * 1000:.3f} ms")

print(f"torch y: {y[:10]}")
print(f"fp16i8 v1 yy1: {yy1[:10]}")
print(f"fp16i8 v2 yy2: {yy2[:10]}")
print(f"fp16i8 v3 yy3: {yy3[:10]}")
print(f"fp32i8 yyy: {yyy[:10]}")

# Compare if yy and y are close enough
if torch.allclose(yy2, y, atol=1e-1):
    print("The results are close enough.")
else:
    print("The results are not close enough.")



'''
rpi5, 4GB. cortexa76 has fp16 native support 
N = 1024
M = int(N * 3.5), 
Execution time for torch_mm8_one: 26.366 ms
Execution time for mm_one_fp16i8    v1: 8.664 ms
Execution time for mm_one_fp16i8    v2: 2.563 ms
Execution time for mm_one_fp16i8    v3: 0.723 ms
Execution time for mm_one_fp32i8: 4.964 ms

'''    