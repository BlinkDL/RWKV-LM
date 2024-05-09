#!/usr/bin/env python3

import torch
import torch.nn as nn


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
    
mod = MyModule()
x1 = mod.state_dict()
x2 = mod.named_parameters()

for name,param in x2:
    print(name)
    # breakpoint()    

# breakpoint()
