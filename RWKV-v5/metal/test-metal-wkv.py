'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code for compiling the custom pytorch extension.

xzl: showcases how metal wkv5 op can be called 

orig code:
"This sample code project is associated with WWDC23 session 10050: [Optimize machine learning for Metal apps](https://developer.apple.com/wwdc23/10050)."

'''

import torch
from torch import nn
import torch.utils.cpp_extension

HEAD_SIZE=64

B=8
T=512
C=768
H=C//HEAD_SIZE

# wraps native code as a py module(?
wkv5_metal = torch.utils.cpp_extension.load(
    name='wkv5',        # xzl: useful??
    sources=['wkv5_op.mm'],
    # verbose=True, 
    extra_cflags=['-std=c++17', '-O3', f"-D_N_={HEAD_SIZE}"],
)

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

def compute_rand_fwd():
    # r=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
    #     memory_format=torch.contiguous_format)
    r=torch.testing.make_tensor((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format,low=0.0,high=1.0)
    k=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    v=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    y=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)

    w=torch.empty((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    u=torch.empty((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    wkv5_metal.forward(B,T,C,H,r,k,v,w,u,y)

def compute_rand_bwd():
    r=torch.testing.make_tensor((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format,low=0.0,high=1.0)
    k=torch.testing.make_tensor((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format,low=0.0,high=1.0)
    v=torch.testing.make_tensor((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format,low=0.0,high=1.0)
    
    w=torch.testing.make_tensor((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format,low=0.0,high=1.0)
    ww=torch.testing.make_tensor((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format,low=0.0,high=1.0)
    u=torch.testing.make_tensor((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format,low=0.0,high=1.0)

    gy=torch.testing.make_tensor((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format,low=0.0,high=1.0)
    gr=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format)
    gk=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format)    
    gv=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format)
    
    gw=torch.empty((B,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format)
    gu=torch.empty((B,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format)    

    breakpoint()
    wkv5_metal.backward(B,T,C,H,r,k,v,w,ww,u,gy,gr,gk,gv,gw,gu)
    breakpoint()

def load_compare_fwd():
    for i in range(10,20):
        fname = f"/tmp/wkv-forwrad-{B}-{T}-{C}-{i}.pth"
        print(f"load {fname}...")

        mydict = torch.load(fname,map_location='cpu')
        # breakpoint()
        r=mydict['r'].to(device=mps_device,memory_format=torch.contiguous_format)
        k=mydict['k'].to(device=mps_device,memory_format=torch.contiguous_format)
        v=mydict['v'].to(device=mps_device,memory_format=torch.contiguous_format)
        w=mydict['w'].to(device=mps_device,memory_format=torch.contiguous_format)
        u=mydict['u'].to(device=mps_device,memory_format=torch.contiguous_format)
        y=mydict['y'].to(device=mps_device,memory_format=torch.contiguous_format)

        ew = (-torch.exp(w.float())).contiguous()
        eew = (torch.exp(ew)).contiguous()
        # eew = eew.to(dtype=torch.bfloat16)

        # -- test if mps exp is sane (against cpu exp), ok -- # 
        '''
        ewc = (-torch.exp(w.cpu().float())).contiguous()
        eewc = (torch.exp(ewc)).contiguous()
        torch.testing.assert_close(eew.cpu(), eewc)
        '''

        yy = torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
            memory_format=torch.contiguous_format)
        # breakpoint()
        wkv5_metal.forward(B,T,C,H,r,k,v,eew,u,yy)

        try: 
            torch.testing.assert_close(y, yy)
        except AssertionError as e:
            print(e)

def load_compare_bwd():
    for i in [13,15,17,19,21,51,53,55,57,59]:
        fname = f"/tmp/wkv-backward-{B}-{T}-{C}-{i}.pth"
        print(f"load {fname}...")

        # load test file
        mydict = torch.load(fname,map_location='cpu')
        # breakpoint()
        r=mydict['r'].to(device=mps_device,memory_format=torch.contiguous_format)
        k=mydict['k'].to(device=mps_device,memory_format=torch.contiguous_format)
        v=mydict['v'].to(device=mps_device,memory_format=torch.contiguous_format)
        
        eew=mydict['eew'].to(device=mps_device,memory_format=torch.contiguous_format)
        ew=mydict['ew'].to(device=mps_device,memory_format=torch.contiguous_format)
        u=mydict['u'].to(device=mps_device,memory_format=torch.contiguous_format)

        gy=mydict['gy'].to(device=mps_device,memory_format=torch.contiguous_format)
        gr=mydict['gr'].to(device=mps_device,memory_format=torch.contiguous_format)
        gk=mydict['gk'].to(device=mps_device,memory_format=torch.contiguous_format)
        gv=mydict['gv'].to(device=mps_device,memory_format=torch.contiguous_format)

        gw=mydict['gw'].to(device=mps_device,memory_format=torch.contiguous_format)
        gu=mydict['gu'].to(device=mps_device,memory_format=torch.contiguous_format)

        # --- output we calc'd  --- #
        gr1=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format)
        gk1=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format)    
        gv1=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format)
        
        gw1=torch.empty((B,C), device=mps_device, dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format)
        gu1=torch.empty((B,C), device=mps_device, dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format)    
        # --------------------------# 

        # breakpoint()
        wkv5_metal.backward(B,T,C,H,r,k,v,eew,ew,u,gy,gr1,gk1,gv1,gw1,gu1)
        torch.testing.assert_close(gr, gr1)
        torch.testing.assert_close(gk, gk1)
        torch.testing.assert_close(gv, gv1)
        torch.testing.assert_close(gw, gw1)
        torch.testing.assert_close(gu, gu1)

if __name__ == "__main__":
    assert(C%HEAD_SIZE==0)
    load_compare_fwd()
    # compute_rand_bwd()
    # load_compare_bwd()