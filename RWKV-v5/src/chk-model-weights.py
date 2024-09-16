#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
# import torch.nn as nn
from torch.nn import functional as F

models=[]

if __name__ == "__main__":
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-0.pth")
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-10.pth")
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-20.pth")
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-30.pth")
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-40.pth")
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-51.pth")
    models.append("/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-61.pth")


    for MODEL in models:
        print(f"load {MODEL}")
        # Step 1: Load the model's state dictionary
        state_dict = torch.load(MODEL)
        # breakpoint()
        # print(state_dict.keys())

        # Step 2: Extract the 'head_l1fc1' parameter tensor
        param_tensor = state_dict['head_l1fc1.weight']
        # Step 3: Calculate the L2 norm
        l2_norm = torch.norm(param_tensor, p=2)
        # Step 4: Print the L2 norm
        print("L2 Norm of 'head_l1fc1':", l2_norm.item())

        # Step 2: Extract the 'head_l1fc1' parameter tensor
        param_tensor = state_dict['head_l1fc2.weight']
        # Step 3: Calculate the L2 norm
        l2_norm = torch.norm(param_tensor, p=2)
        # Step 4: Print the L2 norm
        print("L2 Norm of 'head_l1fc2':", l2_norm.item())
