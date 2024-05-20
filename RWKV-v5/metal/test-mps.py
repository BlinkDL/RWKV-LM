# https://github.com/pytorch/pytorch/issues/78168

import torch

a = torch.rand(1, device='mps')

dtypes = [
    torch.float32, torch.float, torch.float64, torch.double,
    torch.float16, torch.half, torch.bfloat16, torch.complex64,
    torch.complex128, torch.cdouble, torch.uint8, torch.int8,
    torch.int16, torch.short, torch.int32, torch.int, torch.int64,
    torch.long, torch.bool, torch.quint8, torch.qint8, torch.quint4x2
]

valids_dtypes = []
invalids_dtypes = []
for dtype in dtypes:
    try:
        a.type(dtype)
        valids_dtypes.append(dtype)
    except Exception as e:
        invalids_dtypes.append(dtype)
        print(f"[{dtype}]", e)
print()
print("Valid Types:", valids_dtypes)
print("Invalid Types:", invalids_dtypes)

print("---")

valids_methods = []
invalids_methods = []
method_names = ["float", "bfloat16", "double", "half", "cdouble", "short", "byte", "char", "int", "long", "bool"]
for name in method_names:
    try:
        getattr(a, name)()
        valids_methods.append(name)
    except Exception as e:
        invalids_methods.append(name)
        print(f"[{name}]", e)
        
print()
print("Valid Methods:", valids_methods)
print("Invalid Methods:", invalids_methods)