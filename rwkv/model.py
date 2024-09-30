########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from typing import Optional
import numpy as np
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
current_path = os.path.dirname(os.path.abspath(__file__))

########################################################################################################

if os.environ.get('RWKV_JIT_ON') != '0':
    os.environ["RWKV_JIT_ON"] = '1'
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module
    def __nop(ob):
        return ob
    MyFunction = __nop
    MyStatic = __nop

# by xzl 
def is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        # Check for common Raspberry Pi hardware identifiers
        if "raspberry" in cpuinfo or any(model in cpuinfo for model in ["bcm2835", "bcm2836", "bcm2837", "bcm2711"]):
            return True
    except FileNotFoundError:
        # /proc/cpuinfo might not exist on non-Linux systems
        return False
    return False
    
# cf above    
def is_odroid():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        if "odroid" in cpuinfo or any(model in cpuinfo for model in []):
            return True
    except FileNotFoundError:
        # /proc/cpuinfo might not exist on non-Linux systems
        return False
    return False
    
'''
xzl: below implement key ops: 
    wkv
    matmul 
        - mm8_seq  (torch/cuda variants
        - mm8_one    (torch/cuda variants...
        - matmul_float (specialized for fp16...
'''

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    try:
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
            verbose=False,
            extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            is_python_module=False)
        DISABLE_CUBLAS_GEMM = False
    except:
        print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
            verbose=False,
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
            is_python_module=False)
        DISABLE_CUBLAS_GEMM = True

    # xzl: below - invoke custom cuda ops loaded above? 
    #       e.g. torch.ops.rwkv.wkv_forward

    @MyStatic
    def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
        assert 1 * C % min(C, 32) == 0
        assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
        assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=k.dtype)
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
        return y, aa, bb, pp

    # xzl: below - int8 versions of mm (need to reimple this?
    @MyStatic
    def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (B, N)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.empty((B, M), device=w.device, dtype=x.dtype)
        torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
        return y
    @MyStatic
    def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (N,)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.zeros((M,), device=w.device, dtype=torch.float32)
        torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
        return y.to(dtype=x.dtype)
else:
    os.environ["RWKV_CUDA_ON"] = '0'

# xzl: dispatch mm8_seq/one to cuda and "torch" variants (i.e. non cuda
@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

@MyStatic
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

if os.environ.get('RWKV_CUDA_ON') == '1':
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        if w.device.type == 'cuda' and x.dtype == torch.float16:
            B, N, M = x.shape[0], w.shape[0], w.shape[1]
            return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_seq(x, w, mx, rx, my, ry)
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        if w.device.type == 'cuda':
            N, M = w.shape[0], w.shape[1]
            return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_one(x, w, mx, rx, my, ry)
else:
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        return torch_mm8_seq(x, w, mx, rx, my, ry)
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        return torch_mm8_one(x, w, mx, rx, my, ry)

def mm8(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor, rx: torch.Tensor, my: torch.Tensor, ry: torch.Tensor):
    if len(x.shape) == 1:
        return mm8_one(x, w, mx, rx, my, ry)
    return mm8_seq(x, w, mx, rx, my, ry)

# xzl: matmul with optional quant (for mm8 above)
def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        assert a.dtype == b.dtype
        return matmul_float(a, b, output_dtype=output_dtype)
    elif b.dtype == torch.uint8:
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    else:
        raise ValueError("Unsupported dtype")

def matmul_sparsity(a, b):
    if len(a.shape) == 1:
        return torch.sparse.mm(a.unsqueeze(0), b.to_sparse()).squeeze()
    else:
        return torch.sparse.mm(a, b.to_sparse())


# xzl: matmul_float
#       speiclized matmul for CUDA, for fp16, for certain shapes....
if os.environ.get('RWKV_CUDA_ON') == '1' and not DISABLE_CUBLAS_GEMM:
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        if output_dtype is None:
            output_dtype = a.dtype
        if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda':
            if len(a.shape) == 1:
                assert len(b.shape) == 2
                c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
                a = a.unsqueeze(0)
            else:
                assert len(a.shape) == len(b.shape)
                assert len(a.shape) == 2 or len(a.shape) == 3
                # torch.empty((*a.shape[:-1], b.shape[-1])) doesn't work with jit
                if len(a.shape) == 2:
                    c = torch.empty((a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device)
                else:
                    c = torch.empty((a.shape[0], a.shape[1], b.shape[-1]), dtype=output_dtype, device=a.device)
            torch.ops.rwkv.gemm_fp16_cublas(a, b, c)
            return c
        else:
            return (a @ b).to(output_dtype)

else:       # xzl: generic, slow path
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        return (a @ b).to(output_dtype)

# xzl: pytorch on MSFT directX...
if os.environ.get('RWKV_DML_ON') == '1':
    import torch_directml
    print("PyTorch with DirectML Enabled")

########################################################################################################

class RWKV(MyModule):
    def __init__(self, model, strategy, verbose = True, convert_and_save_and_exit = None):
        super().__init__()
        if verbose:
            prxxx = lambda *args, **kwargs: print(*args, **kwargs)
        else:
            prxxx = lambda *args, **kwargs: None

        # xzl: dirty statistics for cls head....
        self.stat_runs = 0    # num of fwd passes run
        self.stat_loaded_cls = 0    # num of cls loaded 
        self.stat_loaded_tokens = 0  # num of token "cols" loaded
        # exec time stats, all in sec
        self.stat_time_fwd = 0.0   # total fwd time
        self.stat_time_att = 0.0   
        self.stat_time_ffn = 0.0   
        self.stat_time_cls = 0.0   

        # xzl: parse strategy... e.g. "cuda fp16"... and apply to layers 
        STRATEGY_REGEX = r"^(?:(?:^|->) *(?:cuda(?::[\d]+)?|cpu|mps|dml) (?:fp(?:16|32)|bf16)(?:i8|i4|i3)?(?: \*[\d]+\+?)? *)+$"
        if not re.match(STRATEGY_REGEX, strategy):
            raise ValueError("Invalid strategy. Please read https://pypi.org/project/rwkv/")

        strategy = ('->'.join([x.strip() for x in strategy.split('->')])).replace('->', ' -> ')
        self.args = types.SimpleNamespace()
        args = self.args
        args.MODEL_NAME = model
        args.strategy_string = strategy

        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid fp16 overflow)
        try:
            self.RESCALE_LAYER = int(os.environ["RWKV_RESCALE_LAYER"]) # !!! NOTE: SEEMS YOU SHOULD SET IT TO 999 (disable) FOR RWKV-MUSIC MODELS !!!
        except:
            self.RESCALE_LAYER = 6 if 'fp16' in strategy else 0
        prxxx(f'RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]} RWKV_CUDA_ON {os.environ["RWKV_CUDA_ON"]} RESCALE_LAYER {self.RESCALE_LAYER}\n')

        # xzl: load model... and convert params (saved as bf16 default) per "strategy"
        args.MODEL_NAME = args.MODEL_NAME.strip()
        if not args.MODEL_NAME.endswith('.pth'):
            args.MODEL_NAME += '.pth'
        prxxx(f'Loading {args.MODEL_NAME} ...')
        with torch.no_grad():
            self.w = torch.load(args.MODEL_NAME, map_location='cpu', weights_only=True) # load model to CPU first
            gc.collect()
            w = self.w

            ALREADY_CONVERTED = False
            if '_strategy' in w:
                ALREADY_CONVERTED = True
                assert convert_and_save_and_exit == None # you should only convert a raw model
                prxxx(f"Converted model: strategy {w['_strategy']}, version {w['_version']}\n")
                assert w['_strategy'] == args.strategy_string # if you are using a new strategy, re-convert the model
                assert float(w['_version']) >= 0.7 # sometimes you should re-convert using latest convert_model.py
                assert w['_rescale_layer'] == self.RESCALE_LAYER # must use same RESCALE_LAYER to avoid mistakes
                del w['_strategy']
                del w['_version']
                del w['_rescale_layer']
            
            args.n_embd = w['emb.weight'].shape[1]
            # xzl: detect dimension, etc. (moved down
            # args.n_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            # args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            args.n_layer = 0
            keys = list(w.keys())
            # xzl: guess model version, 
            self.version = 4
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, layer_id+1)
                if 'ln_x' in x:
                    self.version = max(5, self.version)
                if 'gate.weight' in x:
                    self.version = max(5.1, self.version)
                if int(self.version) == 5 and 'att.time_decay' in x:
                    args.n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            self.version = max(5.2, self.version)
                if 'key1.weight' in x: # xzl
                    self.version = max(5.8, self.version)
                if 'key_diag' in x:  # xzl
                    self.version = max(5.9, self.version)
                if 'ffn.key1.weight' in x:
                    self.version = max(5.94, self.version)
                if 'ffn.key_diag' in x:
                    #self.version = max(5.95, self.version)
                    self.version = max(5.95, self.version)
                if 'time_maa' in x:
                    print("SISIXIXIX")
                    self.version = max(6, self.version)
                if int(self.version) == 6 and 'time_faaaa' in x:
                    args.n_head = w[x].shape[0]
            prxxx(f'Model detected: v{self.version:.2f}')
            
            if self.version in [5.8, 5.9]: # our mod
                # xzl: is this right? 
                args.n_att = w['blocks.0.att.key2.weight'].shape[0] # note: transposed matrix
                args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0] # unchanged
            elif self.version in [5.94, 5.95, 5.96]:
                args.n_att = w['blocks.0.att.key2.weight'].shape[0] # note: transposed matrix
                args.n_ffn = w['blocks.0.ffn.key2.weight'].shape[0]
            else: # official model
                args.n_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
                args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix

            ####################### Compute strategy   
            # xzl: & print out "strategy" for each layer... (NB quant weight, no activation)

            s = [x.strip().split(' ') for x in strategy.split('->')]
            plan = [0] * len(s)
            stream_i = -1       # xzl: stream -- layerwise loading. only DRAM->VRAM. needs mod for storage->DRAM
            stream_count = 0
            to_allocate = args.n_layer + 1
            allocated = 0
            free_slots = 0
            for i in range(len(s)):
                si = s[i]
                si1 = si[1]
                if si1.startswith('fp32'): si[1] = [torch.float]
                elif si1.startswith('fp16'): si[1] = [torch.float16]
                elif si1.startswith('bf16'): si[1] = [torch.bfloat16]
                if si1.endswith('i8'): si[1] += [torch.uint8]
                else: si[1] += [si[1][0]]
                if len(si) > 2:
                    ss = si[2]
                    assert ss.startswith('*')
                    if ss.endswith('+'):
                        plan[i] = int(ss[1:-1])
                        stream_i = i
                    else:
                        plan[i] = int(ss[1:])
                    allocated += plan[i]
                    if allocated >= to_allocate:
                        plan[i] += to_allocate - allocated
                        break
                else:
                    free_slots += 1
            if stream_i < 0:
                if free_slots > 0 and to_allocate > allocated:
                    for i in range(len(s)):
                        if plan[i] == 0:
                            plan[i] = (to_allocate - allocated) // free_slots
                            allocated += plan[i]
                            free_slots -= 1
                if to_allocate > allocated:
                    plan[len(s)-1] += to_allocate - allocated
            else:
                if to_allocate > allocated:
                    stream_count = to_allocate - allocated
                    plan[stream_i] += stream_count
            prxxx(f'Strategy: (total {args.n_layer}+1={args.n_layer+1} layers)')
            for i in range(len(s)):
                ss = s[i]
                if i != stream_i:
                    prxxx(f'* {ss[0]} {str(ss[1]).replace("torch.","")}, store {plan[i]} layers')
                else:
                    prxxx(f'* {ss[0]} {str(ss[1]).replace("torch.","")}, store {plan[i]-stream_count} layers, stream {stream_count} layers')
                plan[i] += (0 if i == 0 else plan[i-1])
            self.strategy = [None] * (args.n_layer + 1)
            strategy = self.strategy
            for n in range(args.n_layer + 1):
                for i in range(len(s)):
                    if n < plan[i]:
                        strategy[n] = types.SimpleNamespace()
                        strategy[n].device = s[i][0]
                        strategy[n].atype = s[i][1][0]
                        strategy[n].wtype = s[i][1][1]
                        strategy[n].stream = False
                        if strategy[n].device == 'dml':
                            strategy[n].device = torch_directml.device()
                        if i == stream_i and n >= (plan[i] - stream_count):
                            strategy[n].stream = True
                        break
                prxxx(f"{n}-{strategy[n].device}-{str(strategy[n].atype).replace('torch.','')}-{str(strategy[n].wtype).replace('torch.','')}{'-stream' if strategy[n].stream else ''}",end=' ')
            prxxx()

            ####################### Load weights to self.w
            # xzl: below - convert weights per layer strategy...
            if not ALREADY_CONVERTED:
                try: # precompute embedding         xzl: fuse layers?? (emb + ln0?
                    w['emb.weight'] = F.layer_norm(w['emb.weight'], (args.n_embd,), weight=w['blocks.0.ln0.weight'], bias=w['blocks.0.ln0.bias'])
                except:
                    w['emb.weight'] = F.layer_norm(w['emb.weight'].float(), (args.n_embd,), weight=w['blocks.0.ln0.weight'].float(), bias=w['blocks.0.ln0.bias'].float())
                del w['blocks.0.ln0.weight']
                del w['blocks.0.ln0.bias']

            print_need_newline = False

            REAL_TIME_FIRST = False
            args.time_state = False
            for x in list(w.keys()):
                if '.time_faaaa' in x: REAL_TIME_FIRST = True
                if '.time_state' in x: args.time_state = True
            if REAL_TIME_FIRST:
                w = {k.replace('.time_faaaa','.time_first') if '.time_faaaa' in k else k: v for k, v in w.items()}
                self.w = w
            
            # xzl: below - convert weights per layer strategy...
            keys = list(w.keys())
            total_parameter_size = 0
            for x in keys:
                parameter_size = 0
                w[x].requires_grad = False
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                if ('ln_out.' in x) or ('head.' in x):
                    layer_id = args.n_layer
                dd = strategy[layer_id]  
                DEVICE = dd.device
                ATYPE = dd.atype
                WTYPE = dd.wtype

                if not ALREADY_CONVERTED:
                    if self.RESCALE_LAYER > 0:  # xzl we didnt touch these..
                        if 'att.output.weight' in x:
                            w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
                        if 'ffn.value.weight' in x:
                            w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))

                    if '.time_' in x:
                        w[x] = w[x].squeeze()
                    if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'gate.weight' in x or 'output.weight' in x or 'head.weight' in x:
                        w[x] = w[x].t()     # xzl transposed (b/c of linear layer
                    if ('head_l1' in x and '.weight' in x) or ('head_l2' in x and '.weight' in x): 
                        w[x] = w[x].t()   # for compressed cls head. same spirit as above. 
                    #xzl: mimic above 
                    if 'key1.weight' in x or 'value1.weight' in x or 'receptance1.weight' in x or 'gate1.weight' in x \
                        or 'key2.weight' in x or 'value2.weight' in x or 'receptance2.weight' in x or 'gate2.weight' in x:
                        w[x] = w[x].t()     # xzl transposed 
                    if '.time_decay' in x and '_w' not in x: # need fp32 for this
                        if self.version == 4:
                            w[x] = -torch.exp(w[x].float())
                        elif int(self.version) == 5:
                            w[x] = torch.exp(-torch.exp(w[x].float())).reshape(-1,1,1)
                            if self.version in [5.2, 5.8, 5.9, 5.94, 5.95, 5.96]:
                                w[x] = w[x].reshape(args.n_head, -1, 1)
                        elif self.version == 6.0:
                            w[x] = w[x].float().reshape(args.n_head, -1, 1)
                    elif '.time_first' in x: # need fp32 for this
                        if self.version == 4:
                            w[x] = w[x].float()
                        elif int(self.version) in [5, 6]:
                            if REAL_TIME_FIRST:
                                w[x] = w[x].float().reshape(-1,1,1)
                            else:
                                w[x] = torch.exp(w[x].float()).reshape(-1,1,1)
                            if self.version in [5.2, 5.8, 5.9, 5.94, 5.95, 5.96, 6.0]:
                                w[x] = w[x].reshape(args.n_head, -1, 1)
                    elif '.ln_x' in x: # need fp32 for group_norm
                        w[x] = w[x].float()
                    else:
                        if (len(w[x].shape) == 2) and ('emb' not in x) and ('_w1' not in x) and ('_w2' not in x):
                            if WTYPE != torch.uint8:  # xzl: (default weight) cast to WTYPE
                                w[x] = w[x].to(dtype=WTYPE)
                            else:   # xzl: cast to torch.uint8, compute min/max, then scale..
                                w[x] = w[x].float()

                                if w[x].shape[0] > w[x].shape[1]:
                                    w[x+'_my'] = torch.amin(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] - w[x+'_my']
                                    w[x+'_mx'] = torch.amin(w[x], dim=0)
                                    w[x] = w[x] - w[x+'_mx']
                                    w[x+'_rx'] = torch.amax(w[x], dim=0)
                                    w[x] = w[x] / w[x+'_rx']
                                    w[x+'_ry'] = torch.amax(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] / w[x+'_ry']
                                else:
                                    w[x+'_mx'] = torch.amin(w[x], dim=0)
                                    w[x] = w[x] - w[x+'_mx']
                                    w[x+'_my'] = torch.amin(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] - w[x+'_my']
                                    w[x+'_rx'] = torch.amax(w[x], dim=0)
                                    w[x] = w[x] / w[x+'_rx']
                                    w[x+'_ry'] = torch.amax(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] / w[x+'_ry']

                                w[x] = torch.clip(torch.floor(w[x] * 256), min=0, max=255).to(dtype=torch.uint8)
                                w[x+'_mx'] = w[x+'_mx'].to(dtype=ATYPE).contiguous()
                                # 16 might be further quantization for storage efficiency
                                w[x+'_rx'] = (w[x+'_rx'] / 16).to(dtype=ATYPE).contiguous()
                                w[x+'_my'] = w[x+'_my'].to(dtype=ATYPE).contiguous()
                                w[x+'_ry'] = (w[x+'_ry'] / 16).to(dtype=ATYPE).contiguous()
                        else:
                            w[x] = w[x].to(dtype=ATYPE)
                
                if convert_and_save_and_exit == None:    # xzl: force weight to be contig in cpu mem... TBD for stream mode
                    if 'emb.' in x:
                        w[x] = w[x].contiguous()
                    elif (dd.stream) and (x.endswith('key.weight') or x.endswith('value.weight') or x.endswith('receptance.weight') or x.endswith('output.weight')):
                        try:
                            w[x] = w[x].contiguous().pin_memory() # if you see "CUDA error: out of memory" here, that's out of CPU RAM, not VRAM. Get more RAM :)
                        except:
                            print('Note: You are running out of RAM. Get more CPU RAM. Now this will run much slower.')
                    elif DEVICE != 'cpu':
                        w[x] = w[x].to(device=DEVICE).contiguous()
                    
                    if (dd.stream) or (DEVICE != 'cpu'):
                        try:
                            w[x+'_mx'] = w[x+'_mx'].to(device=DEVICE).contiguous()
                            w[x+'_rx'] = w[x+'_rx'].to(device=DEVICE).contiguous()
                            w[x+'_my'] = w[x+'_my'].to(device=DEVICE).contiguous()
                            w[x+'_ry'] = w[x+'_ry'].to(device=DEVICE).contiguous()
                        except:
                            pass

                if 'ffn.value.weight' in x:     # xzl: reach the last weight of a layer??? so GC??
                    gc.collect()
                    if 'cuda' in args.strategy_string:
                        torch.cuda.empty_cache()

                # xzl: dump per layer info...
                shape = [i for i in w[x].shape if i != 1]
                nelement = 0

                if len(shape) > 2:
                    nelement = shape[0] * shape[1] * shape[2]
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)} {str(shape[2]).rjust(5)}"
                elif len(shape) > 1:
                    nelement = shape[0] * shape[1]
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}      "
                else:
                    nelement = shape[0]
                    shape = f" {str(shape[0]).rjust(5)}            "

                dt = str(w[x].dtype).replace('torch.', '')
                dt = dt.replace('float32', 'f32').replace('bfloat16', 'bf16').replace('float16', 'f16').replace('uint8', 'i8')

                if dt == "bf16" or dt == "f16":
                    parameter_size = nelement * 2
                elif dt == "f32":
                    parameter_size = nelement * 4
                elif dt == "i8":
                    parameter_size = nelement * 1

                total_parameter_size += parameter_size

                MiB = 1024 * 1024

                if layer_id == 0 or layer_id >= args.n_layer-1:
                    if print_need_newline:
                        prxxx('\n', end = '')
                        print_need_newline = False

                    prxxx(x.ljust(32), dt.rjust(4), str(w[x].device).rjust(8), shape,
                            parameter_size / MiB, ' (pinned)' if w[x].is_pinned() else '')
                else:
                    print_need_newline = True
                    prxxx('.', end = '', flush = True)

            ##### xzl: load & build cls lookup table. do it AFTER all weights are transposed, converted
            # self.head_l2org_weight: List[torch.Tensor] = []
            if 'head_l1.weight' in w: # use compressed cls heads                
                import numpy as np
                args.head_K = 200    # XXX
                # md5sum: 1ba8dc5e...
                if is_raspberry_pi() or is_odroid():
                    args.load_token_cls='/data/models/pi-deployment/rwkv-823-cls.npy'
                else: 
                    # args.load_token_cls='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-pre-x59-8x-cls/from-hpc/rwkv-823-cls.npy'
                    args.load_token_cls='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/from-hpc/rwkv-823-cls.npy'
                
                K=args.head_K
                labels = np.load(args.load_token_cls)
                # idx: cls id, element: list of token_id inside the cls
                clusters = []
                # idx: token_id, element: (cls_id, token id within the cls)
                token2cls = []  
                for i in range(K):
                    clusters.append([])
                for i in range(len(labels)):
                    c = labels[i]
                    token2cls.append((c,len(clusters[c])))
                    clusters[c].append(i)
                
                # self.token2cls = torch.tensor(token2cls, device='cuda')  # unused as of now....
                self.clusters = clusters

                # sanity chk: plot histogram statistics of cluster sizes... 
                '''
                import matplotlib.pyplot as plt
                data = [len(sublist) for sublist in clusters]
                # plt.hist(data, bins=20, edgecolor='black')
                plt.hist(data, bins=range(min(data), max(data) + 2), edgecolor='black')
                plt.title('Histogram of cluster sizes')
                plt.xlabel('Cluster size bins')
                plt.ylabel('# of clusters')
                plt.savefig('cluster-size-histogram.jpg', format='jpg', dpi=300)
                breakpoint()
                '''

                self.clusters_tensor = [] # also save as list of tensors
                # each tensor: 1D (#tokens_per_cls). for tensor computation later .
                for ccc in self.clusters:
                    self.clusters_tensor.append(
                        # torch.tensor(ccc, device='cuda'))
                        torch.tensor(ccc, device='cpu'))   # convert later? 

                # build head_l2, but by splitting the original cls head weights
                self.head_l2org_weight = []

                for cls in range(0, len(clusters)):
                    # NB: w['head.weight'] shape (D,vocab), already transposed
                    orghead = w['head.weight'].T  # orghead shape (vocab, D)
                    idx = torch.tensor(clusters[cls], device=orghead.device)
                    # ww = torch.gather(input=orghead,dim=0,index=idx)
                    ww = orghead[idx]
                    w[f'head_l2org.{cls}.weight'] = ww # save it in dict
                    self.head_l2org_weight.append(ww.t())   # alternatively, save in list  

                # cf rwkv/utils.py generate()
                # XXX move it out of "model", to "pipeline"
                self.occurrence = {}  

                # avoid indexing "w" (the model state dict) inside _retrieve_value3(), which 
                # prevents it from using torch.script... 
                self.head_l1_weight = w['head_l1.weight']
                self.vocab = w['head.weight'].shape[1]   # shape D,vocab                    

            prxxx("parameter size: ", f"{total_parameter_size / MiB:.3f} MB")
            
            if convert_and_save_and_exit:
                w['_strategy'] = args.strategy_string
                w['_rescale_layer'] = self.RESCALE_LAYER
                w['_version'] = '0.7'
                if not convert_and_save_and_exit.endswith('.pth'):
                    convert_and_save_and_exit += '.pth'
                prxxx(f'Saving to {convert_and_save_and_exit}...')
                torch.save(w, convert_and_save_and_exit)
                prxxx(f'Converted and saved. Now this will exit.')
                exit(0)
            
            # xzl: below specialized cuda impl for v5.2 (othrewise fall back to torch??
            if self.version == 5.2 and os.environ["RWKV_CUDA_ON"] == '1':
                HEAD_SIZE = args.n_att // args.n_head
                rwkv5 = load(name="rwkv5", sources=[f"{current_path}/cuda/rwkv5_op.cpp", f"{current_path}/cuda/rwkv5.cu"],
                                verbose=False, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3" if os.name != "nt" else "", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])

                # xzl: whole block in a cuda kernel???
                class RWKV_5(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
                        with torch.no_grad():
                            assert HEAD_SIZE == C // H
                            ctx.B = B
                            ctx.T = T
                            ctx.C = C
                            ctx.H = H
                            assert state.dtype == torch.float32
                            assert w.dtype == torch.float32
                            assert r.is_contiguous()
                            assert k.is_contiguous()
                            assert v.is_contiguous()
                            assert w.is_contiguous()                            
                            assert u.is_contiguous()                            
                            assert state.is_contiguous()

                            y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
                            if r.dtype == torch.bfloat16:
                                rwkv5.forward_bf16(B, T, C, H, state, r, k, v, w, u, y)
                            elif r.dtype == torch.float16:
                                rwkv5.forward_fp16(B, T, C, H, state, r, k, v, w, u, y)
                            elif r.dtype == torch.float32:
                                rwkv5.forward_fp32(B, T, C, H, state, r, k, v, w, u, y)
                            return y, state
                self.RWKV_5 = RWKV_5

            if self.version == 6.0 and os.environ["RWKV_CUDA_ON"] == '1':
                HEAD_SIZE = args.n_att // args.n_head
                rwkv6 = load(name="rwkv6", sources=[f"{current_path}/cuda/rwkv6_op.cpp", f"{current_path}/cuda/rwkv6.cu"],
                                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3" if os.name != "nt" else "", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
                    
                class RWKV_6(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
                        with torch.no_grad():
                            assert HEAD_SIZE == C // H
                            ctx.B = B
                            ctx.T = T
                            ctx.C = C
                            ctx.H = H
                            assert state.dtype == torch.float32
                            assert w.dtype == torch.float32
                            assert r.is_contiguous()
                            assert k.is_contiguous()
                            assert v.is_contiguous()
                            assert w.is_contiguous()
                            assert u.is_contiguous()
                            eew = torch.exp(-torch.exp(w.float())).contiguous()

                            y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
                            if r.dtype == torch.bfloat16:
                                rwkv6.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
                            elif r.dtype == torch.float16:
                                rwkv6.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
                            elif r.dtype == torch.float32:
                                rwkv6.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
                            return y, state
                self.RWKV_6 = RWKV_6
        
            gc.collect()
            if 'cuda' in args.strategy_string:
                torch.cuda.empty_cache()

    def RUN_RWKV_5(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_5.apply(B, T, C, H, state, r, k, v, w, u)

    def RUN_RWKV_6(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_6.apply(B, T, C, H, state, r, k, v, w, u)

    # xzl: below, non-cuda version...
    # XXX_one -- for single input token; XXX_seq -- for a seq of tokens (prompt??
    ########################################################################################################

    # xzl: this 
    @MyFunction
    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    # xzl: ours, based on above
    @MyFunction
    def ffn_one_v5_8(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, 
                     rw1, rw2, # sans rwdiag, 
                     kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1) 
        # r = torch.relu(r) ** 2    // no relu
        r = matmul(r, rw2)
        r = torch.sigmoid(r)

        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    @MyFunction
    def ffn_one_v5_9(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw,
                         rw1, rw2, rwdiag, 
                         kmx, krx, kmy, kry, vmx, vrx, vmy, vry, 
                         rmx1, rrx1, rmy1, rry1,
                         rmx2, rrx2, rmy2, rry2,
                         ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: should use matmul??
        r = torch.sigmoid(r)

        k = matmul(kx, kw, kmx, krx, kmy, kry)

        # true if k <= 0
        #k_zero_mask = (k <= 0)
        #
        #if len(kx.shape) == 1:
        #    kw_related_zero_mask = kx.view(-1, 1) * k_zero_mask.to(torch.float16)
        #else:
        #    kw_related_zero_mask = torch.matmul(kx.t(), k_zero_mask.to(torch.float16))
        #
        #k = matmul_sparsity(kx.to(torch.float32), kw.to(torch.float32)).to(torch.float16)

        vx = torch.relu(k) ** 2     # sparse actiavtion

        # which neuron is activated? if 0 = inactive, else active
        mask = (vx != 0).half()
        #used_weight = kx.unsqueeze(1) @ mask.unsqueeze(0)

        # check # of zeroes
        #num_zeros = torch.sum(vx == 0).item()
        #total_elements = vx.numel()
        #zero_ratio = num_zeros / total_elements
        #print(zero_ratio)


        # sparsification
        flattened = vx.to(torch.float32).view(-1)
        th = torch.quantile(flattened, .9)
        # spartified vx
        #vx = torch.where(vx < th.to(vx.dtype), torch.tensor(0.0, dtype=vx.dtype), vx)

        v = matmul(vx, vw, vmx, vrx, vmy, vry)

        out = r * v
        return x + out, xx, mask

    # xzl: ours, based on above
    @MyFunction
    def ffn_one_v5_94(self, x, sx, ln_w, ln_b, k_mix, r_mix, 
                     kw1, kw2,
                     vw1, vw2,
                     rw1, rw2, rwdiag, 
                     kmx, krx, kmy, kry, vmx, vrx, vmy, vry, 
                     rmx1, rrx1, rmy1, rry1,
                     rmx2, rrx2, rmy2, rry2,
                     ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: should use matmul??
        r = torch.sigmoid(r)

        k = matmul(kx, kw1, kmx, krx, kmy, kry)
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx, krx, kmy, kry)
        
        vx = torch.relu(k) ** 2
        v = matmul(vx, vw1, vmx, vrx, vmy, vry)
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx, vrx, vmy, vry)

        out = r * v
        return x + out, xx
    
    # xzl: ours, based on above
    @MyFunction
    def ffn_one_v5_95(self, x, sx, ln_w, ln_b, k_mix, r_mix, 
                     kw1, kw2, kwdiag,
                     vw1, vw2, vwdiag,
                     rw1, rw2, rwdiag, 
                     kmx, krx, kmy, kry, vmx, vrx, vmy, vry, 
                     rmx1, rrx1, rmy1, rry1,
                     rmx2, rrx2, rmy2, rry2,
                     ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: should use matmul??
        r = torch.sigmoid(r)

        k = matmul(kx, kw1, kmx, krx, kmy, kry)
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx, krx, kmy, kry)
        
        # sol 3.
        k1 = kx @ torch.diag(kwdiag)
        k += F.pad(k1,(0, k.shape[-1] - k1.shape[-1]))
        
        vx = torch.relu(k) ** 2
        v = matmul(vx, vw1, vmx, vrx, vmy, vry)
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx, vrx, vmy, vry)

        # sol 3
        v1 = vx @ torch.diag(vwdiag)
        
        if len(v1.shape) == 1:
            v1_trunc = v1[:v.shape[-1]]
        elif len(v1.shape) == 2:
            v1_trunc = v1[:, :v.shape[-1]]
        else:
            v1_trunc = v1[:, :, :v.shape[-1]]

        v += v1_trunc

        out = r * v
        return x + out, xx

    # xzl: ours, based on above
    @MyFunction
    def ffn_one_v5_96(self, x, sx, ln_w, ln_b, k_mix, r_mix, 
                     kw1, kw2, kwdiag,
                     vw1, vw2, vwdiag,
                     rw1, rw2, rwdiag, 
                     kmx, krx, kmy, kry, vmx, vrx, vmy, vry, 
                     rmx1, rrx1, rmy1, rry1,
                     rmx2, rrx2, rmy2, rry2,
                     ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: should use matmul??
        r = torch.sigmoid(r)

        k = matmul(kx, kw1, kmx, krx, kmy, kry)
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx, krx, kmy, kry)
        
        # sol 1.
        k1 = kx * kwdiag
        k1 = k1.sum(dim=-1, keepdim=True)
        k += k1
        
        vx = torch.relu(k) ** 2
        v = matmul(vx, vw1, vmx, vrx, vmy, vry)
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx, vrx, vmy, vry)

        # sol 1
        v1 = k * vwdiag
        v1 = v1.sum(dim=-1, keepdim=True)
        v += v1

        out = r * v
        return x + out, xx
    
    # xzl: this 
    @MyFunction
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1,:]

    # xzl: ours, based on above
    @MyFunction
    def ffn_seq_v5_8(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, 
                     rw1, rw2, # sans rwdiag, 
                     kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1) 
        # r = torch.relu(r) ** 2    # no relu
        r = matmul(r, rw2)
        r = torch.sigmoid(r)        

        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1,:]
    
    # xzl: ours, based on above
    @MyFunction
    def ffn_seq_v5_9(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw,
                     rw1, rw2, rwdiag, 
                     kmx, krx, kmy, kry, 
                     vmx, vrx, vmy, vry, 
                     rmx1, rrx1, rmy1, rry1,
                     rmx2, rrx2, rmy2, rry2,
                     ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: use matmul??
        r = torch.sigmoid(r)        

        k = matmul(kx, kw, kmx, krx, kmy, kry)

        # true if k <= 0
        #k_zero_mask = (k <= 0)

        #if len(kx.shape) == 1:
        #    kw_related_zero_mask = kx.view(-1, 1) * k_zero_mask.to(torch.float16)
        #else:
        #    kw_related_zero_mask = torch.matmul(kx.t(), k_zero_mask.to(torch.float16))

        #kw[kw_related_zero_mask > 0] = 0

        #k = matmul_sparsity(kx.to(torch.float32), kw.to(torch.float32)).to(torch.float16)

        vx = torch.relu(k) ** 2   # sparse actiavtion

        # check # of zeroes
        #num_zeros = torch.sum(vx == 0).item()
        #total_elements = vx.numel()
        #zero_ratio = num_zeros / total_elements
        #print(zero_ratio)

        # sparsification
        flattened = vx.to(torch.float32).view(-1)
        th = torch.quantile(flattened, .9)
        # spartified vx
        #vx = torch.where(vx < th.to(vx.dtype), torch.tensor(0.0, dtype=vx.dtype), vx)

        v = matmul(vx, vw, vmx, vrx, vmy, vry)

        out = r * v
        return x + out, xx[-1,:], None

    @MyFunction
    def ffn_seq_v5_94(self, x, sx, ln_w, ln_b, k_mix, r_mix, 
                         kw1, kw2,
                         vw1, vw2,
                         rw1, rw2, rwdiag, 
                         kmx, krx, kmy, kry, 
                         vmx, vrx, vmy, vry, 
                         rmx1, rrx1, rmy1, rry1,
                         rmx2, rrx2, rmy2, rry2,
                         ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: use matmul??
        r = torch.sigmoid(r)        

        k = matmul(kx, kw1, kmx, krx, kmy, kry)
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx, krx, kmy, kry)

        vx = torch.relu(k) ** 2
        v = matmul(vx, vw1, vmx, vrx, vmy, vry)
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx, vrx, vmy, vry)

        out = r * v
        return x + out, xx[-1,:]

    @MyFunction
    def ffn_seq_v5_95(self, x, sx, ln_w, ln_b, k_mix, r_mix, 
                         kw1, kw2, kwdiag,
                         vw1, vw2, vwdiag,
                         rw1, rw2, rwdiag, 
                         kmx, krx, kmy, kry, 
                         vmx, vrx, vmy, vry, 
                         rmx1, rrx1, rmy1, rry1,
                         rmx2, rrx2, rmy2, rry2,
                         ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: use matmul??
        r = torch.sigmoid(r)        

        k = matmul(kx, kw1, kmx, krx, kmy, kry)
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx, krx, kmy, kry)
        # sol3.
        k1 = kx @ torch.diag(kwdiag)
        k += F.pad(k1, (0, k.shape[-1] - k1.shape[-1]))
        # sol1.
        #k1 = kx * kwdiag
        #k1 = k1.sum(dim=-1, keepdim=True)
        #k += k1


        vx = torch.relu(k) ** 2
        v = matmul(vx, vw1, vmx, vrx, vmy, vry)
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx, vrx, vmy, vry)

        # sol 3.
        v1 = vx @ torch.diag(vwdiag)
        if len(v1.shape) == 1:
            v1_trunc = v1[:v.shape[-1]]
        elif len(v1.shape) == 2:
            v1_trunc = v1[:, :v.shape[-1]]
        else:
            v1_trunc = v1[:, :, :v.shape[-1]]
        v += v1_trunc

        out = r * v
        return x + out, xx[-1,:]

    @MyFunction
    def ffn_seq_v5_96(self, x, sx, ln_w, ln_b, k_mix, r_mix, 
                         kw1, kw2, kwdiag,
                         vw1, vw2, vwdiag,
                         rw1, rw2, rwdiag, 
                         kmx, krx, kmy, kry, 
                         vmx, vrx, vmy, vry, 
                         rmx1, rrx1, rmy1, rry1,
                         rmx2, rrx2, rmy2, rry2,
                         ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry)) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2)
        r += rx @ torch.diag(rwdiag)   # xzl: use matmul??
        r = torch.sigmoid(r)        

        k = matmul(kx, kw1, kmx, krx, kmy, kry)
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx, krx, kmy, kry)
        # sol1.
        k1 = kx * kwdiag
        k1 = k1.sum(dim=-1, keepdim=True)
        k += k1


        vx = torch.relu(k) ** 2
        v = matmul(vx, vw1, vmx, vrx, vmy, vry)
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx, vrx, vmy, vry)
        # sol1.
        v1 = k * vwdiag
        v1 = v1.sum(dim=-1, keepdim=True)
        v += v1


        out = r * v
        return x + out, xx[-1,:]
    
    @MyFunction
    def ffn_one_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    @MyFunction
    def ffn_seq_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1,:]

    ########################################################################################################

    @MyFunction
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        out = matmul(r * wkv, ow, omx, orx, omy, ory)
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

    @MyFunction
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        T = x.shape[0]
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        out = matmul(r * sx, ow, omx, orx, omy, ory)
        return x + out, xx[-1,:], aa, bb, pp

    ########################################################################################################

    @MyFunction
    def att_one_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    @MyFunction
    def att_seq_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        w = t_decay.reshape(-1, 1)
        u = t_first.reshape(-1, 1)
        ws = w.pow(T).reshape(H, 1, 1)
        ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)
        wk = w.reshape(H, 1, T)
        wb = wk.transpose(-2, -1).flip(1)
        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(H, T, T)

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)

        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v
        
        out = out.transpose(0, 1).contiguous().reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s

    ########################################################################################################

    # xzl: this 
    @MyFunction
    def att_one_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    @MyFunction
    def att_seq_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        w = t_decay.reshape(-1, 1)
        u = t_first.reshape(-1, 1)
        ws = w.pow(T).reshape(H, 1, 1)
        ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)
        wk = w.reshape(H, 1, T)
        wb = wk.transpose(-2, -1).flip(1)
        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(H, T, T)

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v
        
        out = out.transpose(0, 1).contiguous().reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s

    ########################################################################################################

    # xzl: this 
    @MyFunction
    def att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s
    ########################################################################################################
    # xzl: ours, based on att_one_v5_1
    @MyFunction
    def att_one_v5_9(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, 
                     kw1, kw2, kwdiag, vw1, vw2, vwdiag, rw1, rw2, rwdiag, gw1, gw2, gwdiag,   # ours 
                     ow, 
                     kmx1, krx1, kmy1, kry1, 
                     kmx2, krx2, kmy2, kry2, 
                     vmx1, vrx1, vmy1, vry1, 
                     vmx2, vrx2, vmy2, vry2, 
                     rmx1, rrx1, rmy1, rry1, 
                     rmx2, rrx2, rmy2, rry2, 
                     gmx1, grx1, gmy1, gry1, 
                     gmx2, grx2, gmy2, gry2, 
                     omx, orx, omy, ory):        
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]        # xzl: H: head dim? N: # of heads??
        N = x.shape[-1] // H

        # r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)  # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2, output_dtype=torch.float32)     
        r += rx @ torch.diag(rwdiag)   # xzl: should use matmul??
        r = r.view(H,1,N)

        # k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1) # orig
        k = matmul(kx, kw1, kmx1, krx1, kmy1, kry1) 
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx2, krx2, kmy2, kry2, output_dtype=torch.float32)
        k += kx @ torch.diag(kwdiag)
        k = k.view(H,N,1)

        # v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N) # orig
        v = matmul(vx, vw1, vmx1, vrx1, vmy1, vry1) 
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx2, vrx2, vmy2, vry2, output_dtype=torch.float32)     
        v += vx @ torch.diag(vwdiag)
        v = v.view(H,1,N)

        # g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))  @ orig
        g = matmul(gx, gw1, gmx1, grx1, gmy1, gry1)
        g = torch.relu(g) ** 2
        g = matmul(g, gw2, gmx2, grx2, gmy2, gry2) 
        g += gx @ torch.diag(gwdiag)
        g = F.silu(g) 
        
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    # xzl: ours, based on att_one_v5_1
    @MyFunction
    def att_one_v5_8(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, 
                     kw1,kw2, vw1,vw2, rw1,rw2, gw1,gw2,    # ours 
                     ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):        
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]        # xzl: H: head dim? N: # of heads??
        N = x.shape[-1] // H

        # r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)  # orig
        r = matmul(rx, rw1) 
        # r = torch.relu(r) ** 2    # no relu
        r = matmul(r, rw2, output_dtype=torch.float32)     
        r = r.view(H,1,N)

        # k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1) # orig
        k = matmul(kx, kw1) 
        # k = torch.relu(k) ** 2   # no relu
        k = matmul(k, kw2, output_dtype=torch.float32)
        k = k.view(H,N,1)

        # v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N) # orig
        v = matmul(vx, vw1) 
        # v = torch.relu(v) ** 2   # no relu
        v = matmul(v, vw2, output_dtype=torch.float32)     
        v = v.view(H,1,N)

        # g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))  @ orig
        g = matmul(gx, gw1)
        # g = torch.relu(g) ** 2    # no relu
        g = matmul(g, gw2) 
        g = F.silu(g) 
        
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s
    
    # xzl: ours, based on att_seq_v5_2
    @MyFunction
    def att_seq_v5_9(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, 
                     kw1, kw2, kwdiag, vw1, vw2, vwdiag, rw1, rw2, rwdiag, gw1, gw2, gwdiag, 
                     ow, 
                     kmx1, krx1, kmy1, kry1,
                     kmx2, krx2, kmy2, kry2,
                     vmx1, vrx1, vmy1, vry1, 
                     vmx2, vrx2, vmy2, vry2, 
                     rmx1, rrx1, rmy1, rry1, 
                     rmx2, rrx2, rmy2, rry2, 
                     gmx1, grx1, gmy1, gry1, 
                     gmx2, grx2, gmy2, gry2, 
                     omx,  orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1) # orig
        r = matmul(rx, rw1, rmx1, rrx1, rmy1, rry1) 
        r = torch.relu(r) ** 2
        r = matmul(r, rw2, rmx2, rrx2, rmy2, rry2, output_dtype=torch.float32)     
        r += rx @ torch.diag(rwdiag)   # xzl: should use matmul??
        r = r.view(T,H,N).transpose(0, 1)

        # k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0) # orig
        k = matmul(kx, kw1, kmx1, krx1, kmy1, kry1) 
        k = torch.relu(k) ** 2
        k = matmul(k, kw2, kmx2, krx2, kmy2, kry2, output_dtype=torch.float32)     
        k += kx @ torch.diag(kwdiag)
        k = k.view(T,H,N).permute(1, 2, 0)
                
        # v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1) # orig
        v = matmul(vx, vw1, vmx1, vrx1, vmy1, vry1) 
        v = torch.relu(v) ** 2
        v = matmul(v, vw2, vmx2, vrx2, vmy2, vry2, output_dtype=torch.float32)     
        v += vx @ torch.diag(vwdiag)
        v = v.view(T,H,N).transpose(0, 1)

        # g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry)) # orig
        g = matmul(gx, gw1, gmx1, grx1, gmy1, gry1)
        g = torch.relu(g) ** 2
        g = matmul(g, gw2, gmx2, grx2, gmy2, gry2)
        g += gx @ torch.diag(gwdiag)
        g = F.silu(g)

        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s
    
    # xzl: ours, based on att_seq_v5_2
    @MyFunction
    def att_seq_v5_8(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, 
                     kw1,kw2, vw1,vw2, rw1,rw2, gw1,gw2, 
                     ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1) # orig
        r = matmul(rx, rw1) 
        # r = torch.relu(r) ** 2        # no relu
        r = matmul(r, rw2, output_dtype=torch.float32)     
        r = r.view(T,H,N).transpose(0, 1)

        # k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0) # orig
        k = matmul(kx, kw1) 
        # k = torch.relu(k) ** 2        # no relu
        k = matmul(k, kw2, output_dtype=torch.float32)     
        k = k.view(T,H,N).permute(1, 2, 0)
                
        # v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1) # orig
        v = matmul(vx, vw1) 
        # v = torch.relu(v) ** 2        # no relu
        v = matmul(v, vw2, output_dtype=torch.float32)     
        v = v.view(T,H,N).transpose(0, 1)

        # g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry)) # orig
        g = matmul(gx, gw1)
        # g = torch.relu(g) ** 2        # no relu
        g = matmul(g, gw2) 
        g = F.silu(g)

        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s    
    ########################################################################################################

    @MyFunction
    def att_one_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        
        sx = sx - xx
        xxx = xx + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        H = t_decay.shape[0]
        N = x.shape[-1] // H

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
        w = t_decay + (torch.tanh(wx @ td_w1) @ td_w2).float().view(H, N, 1)
        w = torch.exp(-torch.exp(w.float()))

        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + w * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    @MyFunction
    def att_seq_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx
        xxx = xx + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)
        w = torch.exp(-torch.exp(w.float()))
        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + w[t] * s

        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s

    # version 3, ***torchscript friendly***
    #  select the top K cls logits (no sampling) 
    #   NOT tracking cls "frequency" which is not useful to
    #   lm_eval
    #
    #  can be problematic for "chat" output diversity b/c we are NOT sampling cls 
    #  return: token logits (# = vocab)
    from typing import List

    # def _retrieve_value3(self, x, w, head_l1_weight, head_l2org_weight):
    # @MyFunction   # does not matter much???
    def _retrieve_value3_jit(self, x, head_l1_weight, head_l2org_weight: List[torch.Tensor], verbose=False):
        # N=200 # of cls we'll sample
        # N=80 # of cls we'll sample
        N=5 # of cls we'll sample

        t0 = time.time()

        # l1 projection, x: shape (1,D) (regardless of seq_mode
        # x1 = x @ w['head_l1.weight']  # shape (D,#total_cls(200))
        x1 = x @ head_l1_weight
        
        # CLS, CLSPROBS = self.sample_logits(x1, temperature=1.0, top_p=0.85, top_k=N,
        #                          size=N, replace=False)
        
        # --- select, not sampling --- # 
        # "minK" has a high impact on speed... 
        # CLS, CLSPROBS = self.select_logits_jit(x1, minK=3, maxK=100, minProb=.95) # good
        CLS, CLSPROBS, CLS_OTHER, CLSPROBS_OTHER = \
            self.select_logits_jit(x1, minK=3, maxK=100, minProb=.95) # good
        # CLS, CLSPROBS = self.select_logits_jit(x1, minK=5, maxK=40, minProb=.5) 
        # CLS, CLSPROBS, CLS_OTHER, CLSPROBS_OTHER = \
        #     self.select_logits_jit(x1, minK=N, maxK=N, minProb=.5)  # seems quite good? (N=200
            
        t1 = time.time()

        #### now we've picked N cls. L2 projection.... ###
        # vocab = w['head.weight'].shape[1]   # shape D,vocab
        vocab = self.vocab
        # logits = torch.full((vocab,), float("-inf"), device='cuda', dtype=x.dtype) 
        logits = torch.full((vocab,), float("-inf"), device=x.device, dtype=x.dtype) 
        #logits = torch.zeros((vocab,), device='cuda', dtype=x.dtype) 

        t2 = time.time()

        # ---- the "known" logits (from predicted clusters): project x to logits
        # (done) scatter_known_time > proj_known_time (~1.5x-2x), to optimize
        #   idea: since the # of predicted CLS is likely small, we may bundle them in one tensor
        # (with padding), do projection & scatter in one go.
        num_tokens=0
        sum_known_logits_exp = torch.tensor(0.0)  # sum of exp(logits) for all "known" clusters

        proj_known_time = 0.0 
        scatter_known_time = 0.0

        # Collect all indices and corresponding logits
        all_idx = []
        all_x1 = []

        for i in range(0, len(CLS)):
            cls = CLS[i]
            clsprob = CLSPROBS[i]

            tt0 = time.time()

            # x1 = x @ w[f'head_l2org.{cls}.weight'] 
            x1 = x @ head_l2org_weight[cls]
            sum_known_logits_exp += torch.sum(torch.exp(x1)).float()

            tt1 = time.time()

            # cls: cluster id, 
            # self.clusters[cls] list of token_ids in this cls (as scatter idx
            # x: logits over tokens inside cls, (as scatter src
            # idx = torch.tensor(self.clusters[cls], device='cuda')
            idx = self.clusters_tensor[cls]

            num_tokens += idx.shape[0]

            # Collect indices and logits
            all_idx.append(idx)
            all_x1.append(x1)

            proj_known_time += (tt1-tt0)

        # Concatenate all indices and logits
        all_idx = torch.cat(all_idx)
        all_x1 = torch.cat(all_x1)

        # Scatter in one shot
        logits.scatter_(dim=0, index=all_idx, src=all_x1)
        scatter_known_time += (time.time() - tt1)
        
        t3 = time.time()

        # breakpoint()

        # --- the "unknown" logits. fill them with pseduo values  --- #        
        if True:
            scatter_time = 0.0
            cls_log_time = 0.0
            # all "other clusters": pseudo logits 
            Q = sum_known_logits_exp                    
            for i in range(0, len(CLS_OTHER)):
                tt0 = time.time()

                cls = CLS_OTHER[i]
                clsprob = CLSPROBS_OTHER[i]

                # cls: cluster id, 
                # self.clusters[cls] list of token_ids in this cls (as scatter idx
                idx = self.clusters_tensor[cls]
                num_t = idx.shape[0]

                tt1 = time.time()

                # x1: pseudo logits over tokens inside cls, (as scatter src
                # S_j: sum of exp logits for the cluster
                S_j  = Q * (clsprob / CLSPROBS.sum())
                vvv = (S_j / num_t).log()
                # x1 = vvv * torch.ones(num_t, device=x.device, dtype=x.dtype)

                tt2 = time.time()
                # idx: 1D tensor, src: 1D tensor                
                # logits.scatter_(dim=0, index=idx, src=x1)
                logits.index_fill_(dim=0, index=idx, value=vvv)
                tt3 = time.time()

                scatter_time += (tt3-tt2)
                cls_log_time += (tt2-tt1)
                # print(f"cls: {cls}, {tt1-tt0}, {tt2-tt1}, {tt3-tt2}")
            # breakpoint()

        '''
        FL 9/30/24: 
        above: a possible speed up to scatter (in the spirit of computing "known" logits ) would be: 
         1. iterate over all CLS_OTHER, compute pseudo logits
              cat the idx, cat the pseudo logits (as tensors filled with same value)
              e.g. 
              idx_list.append(idx)
              vvv_list.append(torch.full((num_t,), vvv, device=idx.device, dtype=idx.dtype))
         2. scatter them in one go (as a tensor)
              e.g. 
              logits.index_fill_(dim=0, index=idx_cat, value=vvv_cat[0])
         the speed benefit seems insignificant... over index_fill_
        '''
            
        t4 = time.time()

        # -- sanity check: we should have overwritten all prefilled 'inf' ----- #
        assert not torch.isinf(logits).any(), "Tensor logits contains -inf values"

        # -- sanity check: pseudo probs vs. known probs ---- #
        #   accmulated cls probs may diff a bit ... bug or just precision issues??
        #       (bc cls probs are sum of many token probs....)
        if False: 
            pseu_token_probs = torch.softmax(logits, dim=0)                    
            for i in range(0, len(CLS)):
                cls = CLS[i]
                idx = self.clusters_tensor[cls]
                cls_prob = sum(pseu_token_probs[idx])
                print(f"{cls_prob}, {CLSPROBS[i]}")
                # assert(cls_prob == CLSPROBS[i])
            for i in range(0, len(CLS_OTHER)):
                cls = CLS_OTHER[i]
                idx = self.clusters_tensor[cls]
                cls_prob = sum(pseu_token_probs[idx])
                # assert(cls_prob == CLSPROBS_OTHER[i])
            # breakpoint()

        # update statistics
        self.stat_runs += 1
        self.stat_loaded_cls += len(CLS)
        self.stat_loaded_tokens += num_tokens
        
        # -- sanity check ---- ... expensive 
        if False: 
            reallogits = x @ w['head.weight']
            # if N==200:
            if False:
                # if not torch.equal(reallogits,logits):
                # XXX there might be a minor bug somewhere.... causing 
                # minor diff in the two logitgs...
                if not torch.allclose(reallogits,logits, rtol=0.01, atol=0.01):
                    dif=reallogits-logits
                    nzdifmask=dif!=0
                    nzidx=torch.nonzero(nzdifmask, as_tuple=False)
                    breakpoint()
            tokens, probs= self.select_logits(reallogits, 5, 20, 0.85)
            
            # useful CMP: our computed logits (others filled -inf) vs. true logits
            # if K==200, they shall equal 
            if True:
                print(reallogits[tokens])
                print(logits[tokens])
                breakpoint()

        t5 = time.time()
        if verbose:
            print(f"\n cls breakdown time: l1proj {t1-t0:.2f}, logits init {t2-t1:.2f}, logits known {t3-t2:.2f}, logits unknown {t4-t3:.2f}, misc {t5-t4:.2f}")
            print(f"proj_known_time: {proj_known_time:.2f} scatter_known_time: {scatter_known_time:.2f} scatter_unknown_time: {scatter_time:.2f} cls_log_time: {cls_log_time:.2f}")

        return logits 

    ########################################################################################################
    # xzl: cuda versions...
    if os.environ["RWKV_CUDA_ON"] == '1':
        @MyFunction
        def cuda_att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
            T, C = x.shape
            xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
            sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
            kx = xx * k_mix + sx * (1 - k_mix)
            vx = xx * v_mix + sx * (1 - v_mix)
            rx = xx * r_mix + sx * (1 - r_mix)

            r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
            k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            y, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)

            out = matmul(r * y.to(x.dtype), ow, omx, orx, omy, ory)
            return x + out, xx[-1,:], aa, bb, pp

        @MyFunction
        def v5_2_before(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
            xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
            sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
            kx = xx * k_mix + sx * (1 - k_mix)
            vx = xx * v_mix + sx * (1 - v_mix)
            rx = xx * r_mix + sx * (1 - r_mix)
            gx = xx * g_mix + sx * (1 - g_mix)

            r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

            return r, k, v, g, xx[-1,:], s.transpose(-1,-2).contiguous()

        @MyFunction
        def v5_2_after(self, t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            s = s.transpose(-1,-2)
            out = out.reshape(T, H*N)
            out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
            out = out.to(dtype=x.dtype) * g
            out = matmul(out, ow, omx, orx, omy, ory)

            return x + out, xxx, s

        def cuda_att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            r, k, v, g, xxx, ss = self.v5_2_before(x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory)

            out, s = self.RUN_RWKV_5(1, T, self.args.n_att, H, ss, r, k, v, w=t_decay, u=t_first)

            return self.v5_2_after(t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory)

        @MyFunction
        def v6_0_before(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
            sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx
            xxx = xx + sx * x_maa
            xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
            xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            wx = xx + sx * (w_maa + mw)
            kx = xx + sx * (k_maa + mk)
            vx = xx + sx * (v_maa + mv)
            rx = xx + sx * (r_maa + mr)
            gx = xx + sx * (g_maa + mg)

            r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

            w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)

            return r, k, v, g, w, xx[-1,:], s.transpose(-1,-2).contiguous()

        def cuda_att_seq_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            r, k, v, g, w, xxx, ss = self.v6_0_before(x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory)

            out, s = self.RUN_RWKV_6(1, T, self.args.n_att, H, ss, r, k, v, w=w, u=t_first)
            return self.v5_2_after(t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory)

    ########################################################################################################
    # xzl: below-the monolithic forweard func, dispatching to variety of attn, ffn, etc.
    def forward(self, tokens, state, full_output=False):
        with torch.no_grad():
            w = self.w
            args = self.args

            time_measure = {} 
            time_measure['att_dispatch'] = 0
            time_measure['ffn_dispatch'] = 0
            time_measure['att_exec'] = 0
            time_measure['ffn_exec'] = 0
            time_measure['fwd_start'] = time.time()

            # xzl: init state
            if state == None:
                if self.version == 4:
                    state = [None] * args.n_layer * 5
                    for i in range(args.n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
                        dd = self.strategy[i]
                        dev = dd.device
                        atype = dd.atype
                        state[i*5+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                        state[i*5+1] = torch.zeros(args.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                        state[i*5+2] = torch.zeros(args.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                        state[i*5+3] = torch.zeros(args.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e30
                        state[i*5+4] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                elif int(self.version) in [5,6]:
                    state = [None] * args.n_layer * 3
                    for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                        dd = self.strategy[i]
                        dev = dd.device
                        atype = dd.atype
                        state[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                        if args.time_state:
                            state[i*3+1] = w[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
                        else:
                            state[i*3+1] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=dev).contiguous()
                        state[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

            # xzl: seq_mode=True for prompt encoding; =False for autoregression
            seq_mode = len(tokens) > 1

            x = w['emb.weight'][tokens if seq_mode else tokens[0]] # xzl: 'x'-input

            ##### xzl: below- assemble & run layers (each layer)
            #  use custom cuda impl if available, otherwise fall back to torch
            #  XXX: doing this for each token, each layer ... isnt that slow? 
            layer_masks = []
            for i in range(args.n_layer):            
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                dd = self.strategy[i]
                dev = dd.device
                atype = dd.atype
                wtype = dd.wtype

                ############# ---- xzl: below, dispatch ATT ----- #
                time_measure[f'layer{i}_att_dispatch_start'] = time.time()
                if seq_mode:
                    cuda_applicable = os.environ["RWKV_CUDA_ON"] == '1' and 'cuda' in str(dev)
                    if cuda_applicable:
                        ATT = self.cuda_att_seq
                    else:
                        ATT = self.att_seq
                    if self.version == 5:
                        ATT = self.att_seq_v5
                    elif self.version == 5.1:
                        ATT = self.att_seq_v5_1
                    elif self.version == 5.2:
                        ATT = self.att_seq_v5_2
                        if cuda_applicable:
                            ATT = self.cuda_att_seq_v5_2
                    elif self.version == 5.8:
                        ATT = self.att_seq_v5_8
                    elif self.version in [5.9, 5.94, 5.95, 5.96]:
                        ATT = self.att_seq_v5_9
                    elif self.version == 6.0:
                        ATT = self.att_seq_v6_0
                        if cuda_applicable:
                            ATT = self.cuda_att_seq_v6_0

                    FFN = self.ffn_seq
                    if self.version >= 6.0:
                        FFN = self.ffn_seq_v6
                    elif self.version == 5.8:
                        FFN = self.ffn_seq_v5_8
                    elif self.version == 5.9:
                        FFN = self.ffn_seq_v5_9
                    elif self.version == 5.94:
                        FFN = self.ffn_seq_v5_94
                    elif self.version == 5.95:
                        FFN = self.ffn_seq_v5_95
                    elif self.version == 5.96:
                        FFN = self.ffn_seq_v5_96
                else:
                    ATT = self.att_one
                    if self.version == 5:
                        ATT = self.att_one_v5
                    elif self.version == 5.1:
                        ATT = self.att_one_v5_1
                    elif self.version == 5.2:
                        ATT = self.att_one_v5_1 # same as v5.1
                    elif self.version == 5.8:
                        ATT = self.att_one_v5_8
                    elif self.version in [5.9, 5.94, 5.95, 5.96]:
                        ATT = self.att_one_v5_9
                    elif self.version == 6.0:
                        ATT = self.att_one_v6_0
                    
                    FFN = self.ffn_one
                    if self.version >= 6.0:
                        FFN = self.ffn_one_v6
                    elif self.version == 5.8:
                        FFN = self.ffn_one_v5_8
                    elif self.version == 5.9:
                        FFN = self.ffn_one_v5_9
                    elif self.version == 5.94:
                        FFN = self.ffn_one_v5_94
                    elif self.version == 5.95:
                        FFN = self.ffn_one_v5_95
                    elif self.version == 5.96:
                        FFN = self.ffn_one_v5_96

                x = x.to(dtype=atype, device=dev) #xzl:x input

                if self.version in [5.8, 5.9, 5.94, 5.95, 5.96]:
                    kw1 = w[f'{att}key1.weight']
                    kw2 = w[f'{att}key2.weight']                    
                    vw1 = w[f'{att}value1.weight']
                    vw2 = w[f'{att}value2.weight']
                    rw1 = w[f'{att}receptance1.weight']
                    rw2 = w[f'{att}receptance2.weight']                    


                    kmx1 = w[f'{att}key1.weight_mx'] if wtype == torch.uint8 else x
                    krx1 = w[f'{att}key1.weight_rx'] if wtype == torch.uint8 else x
                    kmy1 = w[f'{att}key1.weight_my'] if wtype == torch.uint8 else x
                    kry1 = w[f'{att}key1.weight_ry'] if wtype == torch.uint8 else x

                    kmx2 = w[f'{att}key2.weight_mx'] if wtype == torch.uint8 else x
                    krx2 = w[f'{att}key2.weight_rx'] if wtype == torch.uint8 else x
                    kmy2 = w[f'{att}key2.weight_my'] if wtype == torch.uint8 else x
                    kry2 = w[f'{att}key2.weight_ry'] if wtype == torch.uint8 else x

                    vmx1 = w[f'{att}value1.weight_mx'] if wtype == torch.uint8 else x
                    vrx1 = w[f'{att}value1.weight_rx'] if wtype == torch.uint8 else x
                    vmy1 = w[f'{att}value1.weight_my'] if wtype == torch.uint8 else x
                    vry1 = w[f'{att}value1.weight_ry'] if wtype == torch.uint8 else x

                    vmx2 = w[f'{att}value2.weight_mx'] if wtype == torch.uint8 else x
                    vrx2 = w[f'{att}value2.weight_rx'] if wtype == torch.uint8 else x
                    vmy2 = w[f'{att}value2.weight_my'] if wtype == torch.uint8 else x
                    vry2 = w[f'{att}value2.weight_ry'] if wtype == torch.uint8 else x

                    rmx1 = w[f'{att}receptance1.weight_mx'] if wtype == torch.uint8 else x
                    rrx1 = w[f'{att}receptance1.weight_rx'] if wtype == torch.uint8 else x
                    rmy1 = w[f'{att}receptance1.weight_my'] if wtype == torch.uint8 else x
                    rry1 = w[f'{att}receptance1.weight_ry'] if wtype == torch.uint8 else x

                    rmx2 = w[f'{att}receptance2.weight_mx'] if wtype == torch.uint8 else x
                    rrx2 = w[f'{att}receptance2.weight_rx'] if wtype == torch.uint8 else x
                    rmy2 = w[f'{att}receptance2.weight_my'] if wtype == torch.uint8 else x
                    rry2 = w[f'{att}receptance2.weight_ry'] if wtype == torch.uint8 else x

                    if self.version in [5.9, 5.94, 5.95, 5.96]:
                        kwdiag = w[f'{att}key_diag']
                        vwdiag = w[f'{att}value_diag']
                        rwdiag = w[f'{att}receptance_diag']
                else: 
                    kw = w[f'{att}key.weight']
                    vw = w[f'{att}value.weight']
                    rw = w[f'{att}receptance.weight']

                    # xzl: below, dequant int8 weight (why "else x?"
                    kmx = w[f'{att}key.weight_mx'] if wtype == torch.uint8 else x
                    krx = w[f'{att}key.weight_rx'] if wtype == torch.uint8 else x
                    kmy = w[f'{att}key.weight_my'] if wtype == torch.uint8 else x
                    kry = w[f'{att}key.weight_ry'] if wtype == torch.uint8 else x
                    vmx = w[f'{att}value.weight_mx'] if wtype == torch.uint8 else x
                    vrx = w[f'{att}value.weight_rx'] if wtype == torch.uint8 else x
                    vmy = w[f'{att}value.weight_my'] if wtype == torch.uint8 else x
                    vry = w[f'{att}value.weight_ry'] if wtype == torch.uint8 else x
                    rmx = w[f'{att}receptance.weight_mx'] if wtype == torch.uint8 else x
                    rrx = w[f'{att}receptance.weight_rx'] if wtype == torch.uint8 else x
                    rmy = w[f'{att}receptance.weight_my'] if wtype == torch.uint8 else x
                    rry = w[f'{att}receptance.weight_ry'] if wtype == torch.uint8 else x

                ow = w[f'{att}output.weight']
                omx = w[f'{att}output.weight_mx'] if wtype == torch.uint8 else x
                orx = w[f'{att}output.weight_rx'] if wtype == torch.uint8 else x
                omy = w[f'{att}output.weight_my'] if wtype == torch.uint8 else x
                ory = w[f'{att}output.weight_ry'] if wtype == torch.uint8 else x

                # xzl: intended to move tensor DRAM->VRAM 
                if dd.stream:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)

                if self.version in [5.1, 5.2, 6.0]:
                    gw = w[f'{att}gate.weight']
                    if dd.stream:
                        gw = gw.to(device=dev, non_blocking=True)
                    gmx = w[f'{att}gate.weight_mx'] if wtype == torch.uint8 else x
                    grx = w[f'{att}gate.weight_rx'] if wtype == torch.uint8 else x
                    gmy = w[f'{att}gate.weight_my'] if wtype == torch.uint8 else x
                    gry = w[f'{att}gate.weight_ry'] if wtype == torch.uint8 else x
                elif self.version in [5.8, 5.9, 5.94, 5.95, 5.96]:
                    gw1 = w[f'{att}gate1.weight']
                    gw2 = w[f'{att}gate2.weight']
                    if self.version in [5.9, 5.94, 5.95, 5.96]:
                        gwdiag = w[f'{att}gate_diag']
                    
                    gmx1 = w[f'{att}gate1.weight_mx'] if wtype == torch.uint8 else x
                    grx1 = w[f'{att}gate1.weight_rx'] if wtype == torch.uint8 else x
                    gmy1 = w[f'{att}gate1.weight_my'] if wtype == torch.uint8 else x
                    gry1 = w[f'{att}gate1.weight_ry'] if wtype == torch.uint8 else x

                    gmx2 = w[f'{att}gate2.weight_mx'] if wtype == torch.uint8 else x
                    grx2 = w[f'{att}gate2.weight_rx'] if wtype == torch.uint8 else x
                    gmy2 = w[f'{att}gate2.weight_my'] if wtype == torch.uint8 else x
                    gry2 = w[f'{att}gate2.weight_ry'] if wtype == torch.uint8 else x

                ############# --- xzl: below, run ATT (one or seq) --- # 
                time_measure[f'layer{i}_att_exec_start'] = time.time()
                if self.version == 4:
                    x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
                        x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3],
                        w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                        w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'],
                        w[f'{att}time_decay'], w[f'{att}time_first'],
                        kw, vw, rw, ow,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                        omx, orx, omy, ory,
                        )
                elif self.version == 5:
                    x, state[i*3+0], state[i*3+1] = ATT(
                        x, state[i*3+0], state[i*3+1],
                        w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                        w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
                        w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'],
                        w[f'{att}time_decay'], w[f'{att}time_first'],
                        kw, vw, rw, ow,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                        omx, orx, omy, ory,
                        )
                elif self.version in [5.1, 5.2]:                    
                    x, state[i*3+0], state[i*3+1] = ATT(
                        x, state[i*3+0], state[i*3+1],
                        w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                        w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
                        w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'], w[f'{att}time_mix_g'],
                        w[f'{att}time_decay'], w[f'{att}time_first'],
                        kw, vw, rw, gw, ow,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                        gmx, grx, gmy, gry,
                        omx, orx, omy, ory,
                        )
                elif self.version in [5.8]:
                    x, state[i*3+0], state[i*3+1] = ATT(
                        x, state[i*3+0], state[i*3+1],
                        w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                        w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
                        w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'], w[f'{att}time_mix_g'],
                        w[f'{att}time_decay'], w[f'{att}time_first'],
                        # kw, vw, rw, gw, 
                        kw1,kw2, vw1,vw2, rw1,rw2, gw1,gw2, # sans "diag"
                        ow,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                        gmx, grx, gmy, gry,
                        omx, orx, omy, ory,
                        )                    
                elif self.version in [5.9, 5.94, 5.95, 5.96]:
                    x, state[i*3+0], state[i*3+1] = ATT(
                        x, state[i*3+0], state[i*3+1],
                        w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                        w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
                        w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'], w[f'{att}time_mix_g'],
                        w[f'{att}time_decay'], w[f'{att}time_first'],
                        # kw, vw, rw, gw, 
                        kw1,kw2,kwdiag,vw1,vw2,vwdiag,rw1,rw2,rwdiag,gw1,gw2,gwdiag, 
                        ow,
                        kmx1, krx1, kmy1, kry1,
                        kmx2, krx2, kmy2, kry2,
                        vmx1, vrx1, vmy1, vry1,
                        vmx2, vrx2, vmy2, vry2,
                        rmx1, rrx1, rmy1, rry1,
                        rmx2, rrx2, rmy2, rry2,
                        gmx1, grx1, gmy1, gry1,
                        gmx2, grx2, gmy2, gry2,
                        omx, orx, omy, ory,
                        )
                elif self.version == 6.0:
                    x, state[i*3+0], state[i*3+1] = ATT(
                        x, state[i*3+0], state[i*3+1],
                        w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                        w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
                        w[f'{att}time_maa_x'], w[f'{att}time_maa_w'], w[f'{att}time_maa_k'], w[f'{att}time_maa_v'], w[f'{att}time_maa_r'], w[f'{att}time_maa_g'],
                        w[f'{att}time_maa_w1'], w[f'{att}time_maa_w2'], w[f'{att}time_decay_w1'], w[f'{att}time_decay_w2'],
                        w[f'{att}time_decay'], w[f'{att}time_first'],
                        kw, vw, rw, gw, ow,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                        gmx, grx, gmy, gry,
                        omx, orx, omy, ory,
                        )
                if dd.stream:       # xzl: release VRAM 
                    del kw, vw, rw, ow
                    if self.version in [5.1, 5.2, 6.0]:
                        del gw

                ############# ---- xzl: below, dispatch FFN ----- #
                time_measure[f'layer{i}_ffn_dispatch_start'] = time.time()
                if self.version in [5.8, 5.9, 5.94, 5.95, 5.96]:
                    if self.version in [5.94, 5.95, 5.96]:
                        kw1 = w[f'{ffn}key1.weight']
                        kw2 = w[f'{ffn}key2.weight']
                        vw1 = w[f'{ffn}value1.weight']
                        vw2 = w[f'{ffn}value2.weight']
                    else:
                        kw = w[f'{ffn}key.weight']
                        vw = w[f'{ffn}value.weight']

                    # zero out unimportant layers
                    #kw_mask_indice = np.load(f"unimpt_layers/{i}_layer.npy")
                    #kw[:, kw_mask_indice] = 0

                    rw1 = w[f'{ffn}receptance1.weight']
                    rw2 = w[f'{ffn}receptance2.weight']

                    rmx1 = w[f'{ffn}receptance1.weight_mx'] if wtype == torch.uint8 else x
                    rrx1 = w[f'{ffn}receptance1.weight_rx'] if wtype == torch.uint8 else x
                    rmy1 = w[f'{ffn}receptance1.weight_my'] if wtype == torch.uint8 else x
                    rry1 = w[f'{ffn}receptance1.weight_ry'] if wtype == torch.uint8 else x

                    rmx2 = w[f'{ffn}receptance2.weight_mx'] if wtype == torch.uint8 else x
                    rrx2 = w[f'{ffn}receptance2.weight_rx'] if wtype == torch.uint8 else x
                    rmy2 = w[f'{ffn}receptance2.weight_my'] if wtype == torch.uint8 else x
                    rry2 = w[f'{ffn}receptance2.weight_ry'] if wtype == torch.uint8 else x

                    if self.version in [5.9, 5.94, 5.95, 5.96]:
                        rwdiag = w[f'{ffn}receptance_diag']
                    if self.version in [5.95, 5.96]:
                        kwdiag = w[f'{ffn}key_diag']
                        vwdiag = w[f'{ffn}value_diag']

                else: 
                    kw = w[f'{ffn}key.weight']
                    vw = w[f'{ffn}value.weight']
                    rw = w[f'{ffn}receptance.weight']
                    rmx = w[f'{ffn}receptance.weight_mx'] if wtype == torch.uint8 else x
                    rrx = w[f'{ffn}receptance.weight_rx'] if wtype == torch.uint8 else x
                    rmy = w[f'{ffn}receptance.weight_my'] if wtype == torch.uint8 else x
                    rry = w[f'{ffn}receptance.weight_ry'] if wtype == torch.uint8 else x
                if dd.stream:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                kmx = w[f'{ffn}key.weight_mx'] if wtype == torch.uint8 else x
                krx = w[f'{ffn}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = w[f'{ffn}key.weight_my'] if wtype == torch.uint8 else x
                kry = w[f'{ffn}key.weight_ry'] if wtype == torch.uint8 else x
                vmx = w[f'{ffn}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = w[f'{ffn}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = w[f'{ffn}value.weight_my'] if wtype == torch.uint8 else x
                vry = w[f'{ffn}value.weight_ry'] if wtype == torch.uint8 else x
                if self.version == 4:
                    offset = i*5+4
                elif int(self.version) in [5,6]:
                    offset = i*3+2

                ############# ---- xzl: below, run FFN ----- #
                time_measure[f'layer{i}_ffn_exec_start'] = time.time()
                if self.version in [5.9]:
                    x, state[offset], mask = FFN(
                        x, state[offset],
                        w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                        w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
                        kw, vw,
                        rw1, rw2, rwdiag, 
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx1, rrx1, rmy1, rry1,
                        rmx2, rrx2, rmy2, rry2,
                        )
                    layer_masks.append(mask)
                elif self.version in [5.94]:
                    x, state[offset] = FFN(
                        x, state[offset],
                        w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                        w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
                        kw1, kw2,
                        vw1, vw2,
                        rw1, rw2, rwdiag, 
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx1, rrx1, rmy1, rry1,
                        rmx2, rrx2, rmy2, rry2,
                        )
                elif self.version in [5.95, 5.96]:
                    x, state[offset] = FFN(
                        x, state[offset],
                        w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                        w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
                        kw1, kw2, kwdiag,
                        vw1, vw2, vwdiag,
                        rw1, rw2, rwdiag, 
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx1, rrx1, rmy1, rry1,
                        rmx2, rrx2, rmy2, rry2,
                        )

                elif self.version in [5.8]:
                    x, state[offset] = FFN(
                        x, state[offset],
                        w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                        w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
                        kw, vw, 
                        # rw,
                        rw1, rw2, # sans diag 
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,                    
                        )                    
                elif self.version < 6.0:
                    x, state[offset] = FFN(
                        x, state[offset],
                        w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                        w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
                        kw, vw, rw,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,                    
                        )    
                else:
                    x, state[offset] = FFN(
                        x, state[offset],
                        w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                        w[f'{ffn}time_maa_k'], w[f'{ffn}time_maa_r'],
                        kw, vw, rw,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,                    
                        )
                if dd.stream:                
                    del kw, vw, rw
                
                if self.RESCALE_LAYER > 0:
                    if (i+1) % self.RESCALE_LAYER == 0:
                        x = x / 2
                time_measure[f'layer{i}_ffn_exec_end'] = time.time()

                time_measure['att_exec'] += \
                    time_measure[f'layer{i}_ffn_exec_start'] - time_measure[f'layer{i}_att_exec_start']
                time_measure['ffn_exec'] += \
                    time_measure[f'layer{i}_ffn_exec_end'] - time_measure[f'layer{i}_ffn_exec_start']

            dd = self.strategy[args.n_layer]
            # xzl: below, take last token ONLY even if seq_mode==True, 
            # means that prompt stage only update state. no need to materialize
            # the tokens 
            # "full_output" (default False) seems for debugging 
            #       i.e. materialize tokens even for the prompt stage
            x = x[-1,:] if (seq_mode and (not full_output)) else x
            x = x.to(dtype=dd.atype, device=dd.device)
            
            ############# xzl: all layers done 
            ############# xzl: below: layer norm, cls head...
            time_measure['cls_start'] = time.time()
            x = F.layer_norm(x, (args.n_embd,), weight=w['ln_out.weight'], bias=w['ln_out.bias'])

            if 'head_l1.weight' in w: # use compressed cls heads
                # version 0
                # sample top1 cls. 
                def _retrieve_value0(x, w):
                    '''
                    Current design: greedy sampling cls (L1).
                    cal logits over all clusters. find the cls with highest logit
                    (greedy); within this cls, compute logits over tokens 
                    return: computed logits (for tokens for cls); -inf for other
                    tokens

                    alternatively: TBD
                    cal logits over all clustres, return to the caller
                    the caller: sample a cluster (non greedy). the model: cal logits
                    within that cluster. caller: sample a token. 
                    '''

                    args.alpha_frequency = 0.25
                    args.alpha_presence = 0.25
                    args.alpha_decay = 0.996 # gradually decay the penalty

                    # x: shape D (regardless of seq_mode
                    # l1 projection
                    x1 = x @ w['head_l1.weight']  # shape D,K

                    # cls = x1.argmax(dim=-1)       # greedy sampling. bad results
                    # below: sample 1 cls (i.e. get one cls out of all)
                    # sample_logits (cf test-rwkv-chat.py
                    for n in self.occurrence:       # penalize frequent cls (by reducing their logits...
                        x1[n] -= (args.alpha_presence + self.occurrence[n] * args.alpha_frequency)
                    cls, _ = self.sample_logits(x1, temperature=1.0, top_p=0.7, top_k=100, size=1) 
                    cls = cls[0]
                    # below: update "occrruence" statistics
                    for xxx in self.occurrence:
                        self.occurrence[xxx] *= args.alpha_decay
                    www = 1
                    if cls not in self.occurrence:
                        self.occurrence[cls] = www
                    else:
                        self.occurrence[cls] += www

                    #### now we've picked a cls. compute logits for all tokens within the cls
                    # print(f">>>>>> # self.occurrence = {len(self.occurrence)}")
                    # print(f"\t\t\t\t\t cls {cls} occur {self.occurrence[cls]:.2f} #tokens {len(self.clusters[cls])}")

                    # l2 project x to: logits over all possible tokens within the cls
                    x = x @ w[f'head_l2.{cls}.weight']
                                    
                    vocab = w['head.weight'].shape[1]   # shape D,vocab
                    # cls: cluster id, 
                    # self.clusters[cls] list of token_ids in this cls (as scatter idx
                    # x: logits over tokens inside cls, (as scatter src
                    idx = torch.tensor(self.clusters[cls], device='cuda')
                    # res: logits over all tokens (vocab). prefilled with -inf, used
                    #   as scatter dest
                    res = torch.full((vocab,),float('-inf'),device='cuda',dtype=x.dtype) \
                        .scatter_(0, idx, x)
                    return res

                # version 1: sample N cls
                # return: token logits (# = vocab)
                def _retrieve_value1(x, w):
                    args.alpha_frequency = 0.01     #.25
                    args.alpha_presence = 0.01      #.25
                    args.alpha_decay = 0.996 # gradually decay the penalty

                    # N=100 # of cls we'll sample
                    N=10 # of cls we'll sample

                    # clsprob = .5 # idea: cumulatiev pros for the cls we'll sample

                    # x: shape D (regardless of seq_mode
                    # l1 projection
                    x1 = x @ w['head_l1.weight']  # shape D,K
                    # below: sample N cls (i.e. get N cls out of all)
                    # sample_logits (cf test-rwkv-chat.py

                    for n in self.occurrence:       # penalize frequent cls (by reducing their logits...
                        x1[n] -= (args.alpha_presence + self.occurrence[n] * args.alpha_frequency)
                    
                    CLS, CLSPROBS = self.sample_logits(x1, temperature=1.0, top_p=0.85, top_k=N,
                                             size=N, replace=False)                                         
                    # breakpoint()                    

                    # below: update "occrruence" statistics
                    for xxx in self.occurrence:
                        self.occurrence[xxx] *= args.alpha_decay
                    www = 1
                    for cls in CLS:
                        if cls not in self.occurrence:
                            self.occurrence[cls] = www
                        else:
                            self.occurrence[cls] += www

                    #### now we've picked N cls. L2 projection.... ###

                    vocab = w['head.weight'].shape[1]   # shape D,vocab                
                    probs = torch.full((vocab,), 0.0, device='cuda', dtype=x.dtype)

                    # project x to: probs over all possible tokens within each cls, 
                    #           scaled by the cls prob
                    ntokens=0
                    for i in range(0, len(CLS)):
                        cls = CLS[i]
                        clsprob = CLSPROBS[i]
                        x1 = x @ w[f'head_l2.{cls}.weight']
                        x1 = F.softmax(x1, dim=-1)
                        x1 = x1 * clsprob 
                        ntokens += x1.shape[0]
                        # cls: cluster id, 
                        # self.clusters[cls] list of token_ids in this cls (as scatter idx
                        # x: logits over tokens inside cls, (as scatter src
                        idx = torch.tensor(self.clusters[cls], device='cuda')
                        # breakpoint()    
                        probs.scatter_(dim=0, index=idx, src=x1)

                    if sum(CLSPROBS) < 0.3: 
                        # print(f" CLSPROBS {CLSPROBS} ntokens {ntokens}")
                        pass
                        # idea: maybe we should fallback....
                        breakpoint()

                    # breakpoint()
                    # invert all token probs to (fake) token logits, as expected by pipeline caller
                    logits = torch.log(probs) - torch.logsumexp(torch.log(probs), dim=0)
                    # NB: the probs calculated from these logits may != the probs
                    # computed above, b/c probs from above do not add up to 1 (i.e. unsampled cls will have 
                    # prob of 0
                    return logits

                # version 2
                # return: token logits (# = vocab)
                def _retrieve_value2(x, w):
                    # orig
                    # args.alpha_frequency = 0.25  # orig
                    # args.alpha_presence = 0.25    # orig

                    # need to test these
                    args.alpha_frequency = 0.2  
                    args.alpha_presence = 0.2   

                    # -- kinda repeatitve ...
                    # args.alpha_frequency = 0.1
                    # args.alpha_presence = 0.1

                    args.alpha_decay = 0.996 # gradually decay the penalty

                    # N=200 # of cls we'll sample
                    # N=80 # of cls we'll sample
                    N=5 # of cls we'll sample

                    # clsprob = .5 # idea: cumulatiev pros for the cls we'll sample

                    # x: shape D (regardless of seq_mode
                    # l1 projection
                    x1 = x @ w['head_l1.weight']  # shape D,K
                    # below: sample N cls (i.e. get N cls out of all)
                    # sample_logits (cf test-rwkv-chat.py
                    ## penalize frequent cls (by reducing their logits...  XXXX should do this???
                    for n in self.occurrence:       
                        x1[n] -= (args.alpha_presence + self.occurrence[n] * args.alpha_frequency)
                    
                    # CLS, CLSPROBS = self.sample_logits(x1, temperature=1.0, top_p=0.85, top_k=N,
                    #                          size=N, replace=False)
                    
                    # --- select, not sampling --- # 
                    CLS, CLSPROBS = self.select_logits(x1, minK=5, maxK=100, minProb=.75) 
                    # CLS, CLSPROBS = self.select_logits(x1, minK=5, maxK=40, minProb=.5) 
                    # CLS, CLSPROBS = self.select_logits(x1, minK=N, maxK=N, minProb=.5)  # seems quite good? (N=200

                    # below: update "occrruence" statistics
                    for xxx in self.occurrence:
                        self.occurrence[xxx] *= args.alpha_decay
                    www = 1
                    for cls in CLS:
                        if cls not in self.occurrence:
                            self.occurrence[cls] = www
                        else:
                            self.occurrence[cls] += www

                    #### now we've picked N cls. L2 projection.... ###

                    vocab = w['head.weight'].shape[1]   # shape D,vocab                
                    logits = torch.full((vocab,), float('-inf'), device='cuda', dtype=x.dtype) 

                    # project x to: probs over all possible tokens within each cls, 
                    #           scaled by the cls prob
                    ntokens=0
                    for i in range(0, len(CLS)):
                        cls = CLS[i]
                        clsprob = CLSPROBS[i]
                        x1 = x @ w[f'head_l2org.{cls}.weight'] 

                        ntokens += x1.shape[0]
                        # cls: cluster id, 
                        # self.clusters[cls] list of token_ids in this cl s (as scatter idx
                        # x: logits over tokens inside cls, (as scatter src
                        idx = torch.tensor(self.clusters[cls], device='cuda')
                        #  ------ sanity check: if we use the org head.weight ------ # 
                        if False:
                            yyy=w[f'head_l2org.{cls}.weight']
                            zzz=w['head.weight']
                            for ii in range(len(idx)): 
                                tokenid = idx[ii]
                                if not torch.equal(yyy[:,ii], zzz[:,tokenid]): 
                                    breakpoint()  
                        # print("all good")                        
                        # ---------------------------- # 

                        # since we use the orig head weights, 
                        #   it's ok to concat the raw logits from multi clusters
                        logits.scatter_(dim=0, index=idx, src=x1)
             
                    # -- sanity check ---- ... expensive 
                    if False: 
                        reallogits = x @ w['head.weight']
                        if N==200:
                            # if not torch.equal(reallogits,logits):
                            # XXX there might be a minor bug somewhere.... causing 
                            # minor diff in the two logitgs...
                            if not torch.allclose(reallogits,logits, rtol=0.01, atol=0.01):
                                dif=reallogits-logits
                                nzdifmask=dif!=0
                                nzidx=torch.nonzero(nzdifmask, as_tuple=False)
                                breakpoint()
                        tokens, probs= self.select_logits(reallogits, 5, 20, 0.85)
                        
                        # useful CMP: our computed logits (others filled -inf) vs. true logits
                        # if K==200, they shall equal 
                        if True:
                            print(reallogits[tokens])
                            print(logits[tokens])
                            breakpoint()

                    return logits

                # version 3. select the top K cls logits (no sampling) 
                #   NOT tracking cls "frequency" which is not useful to
                #   lm_eval
                #
                #  can be problematic for "chat" output diversity b/c we are NOT sampling cls 
                #  return: token logits (# = vocab)
                # @MyStatic
                def _retrieve_value3(x, w):
                    # N=200 # of cls we'll sample
                    # N=80 # of cls we'll sample
                    N=5 # of cls we'll sample

                    # x: shape D (regardless of seq_mode
                    # l1 projection
                    x1 = x @ w['head_l1.weight']  # shape D,K
                    
                    # CLS, CLSPROBS = self.sample_logits(x1, temperature=1.0, top_p=0.85, top_k=N,
                    #                          size=N, replace=False)
                    
                    # --- select, not sampling --- # 
                    # "minK" has a high impact on speed... 
                    # CLS, CLSPROBS = self.select_logits(x1, minK=3, maxK=100, minProb=.95) # good
                    CLS, CLSPROBS, CLS_OTHER, CLSPROBS_OTHER = \
                        self.select_logits(x1, minK=3, maxK=100, minProb=.95) # good
                    # CLS, CLSPROBS = self.select_logits(x1, minK=5, maxK=40, minProb=.5) 
                    # CLS, CLSPROBS, CLS_OTHER, CLSPROBS_OTHER = \
                    #     self.select_logits(x1, minK=N, maxK=N, minProb=.5)  # seems quite good? (N=200

                    # find minimum
                    # total_cls, total_clsprobs = self.select_logits(x1, minK=200, maxK=200, minProb=0)
                    # min_cls = total_cls[-1]

                    # min_x1 = x @ w[f'head_l2org.{min_cls}.weight']
                    # min_logit = torch.min(min_x1)
                        
                    # WC: the lowest cls prob does not mean that the cluster has 
                    # the lowest logit. The code below is to check this
                    if False:
                        x_min_logit = 0
                        for i in range(0, len(total_cls)):
                            cls = total_cls[i]
                            x1 = x @ w[f'head_l2org.{cls}.weight'] 
                            temp = torch.min(x1)
                            if temp <= x_min_logit:
                                x_min_logit = temp
                                print("check")
                                print(cls)
                                print(temp)
                                print(total_clsprobs[i])
                        print(x_min_logit)
                        print(min_logit)
                        
                    #### now we've picked N cls. L2 projection.... ###
                    vocab = w['head.weight'].shape[1]   # shape D,vocab                
                    # logits = torch.full((vocab,), float("-inf"), device='cuda', dtype=x.dtype) 
                    logits = torch.full((vocab,), float("-inf"), device=x.device, dtype=x.dtype) 
                    #logits = torch.zeros((vocab,), device='cuda', dtype=x.dtype) 

                    num_tokens=0
                    # all "known clusters": project x to logits
                    sum_known_logits_exp = 0  # sum of exp(logits) for all "known" clusters

                    for i in range(0, len(CLS)):
                        cls = CLS[i]
                        clsprob = CLSPROBS[i]
                        x1 = x @ w[f'head_l2org.{cls}.weight'] 
                        sum_known_logits_exp += torch.sum(torch.exp(x1)).float()

                        # cls: cluster id, 
                        # self.clusters[cls] list of token_ids in this cls (as scatter idx
                        # x: logits over tokens inside cls, (as scatter src
                        # idx = torch.tensor(self.clusters[cls], device='cuda')
                        idx = self.clusters_tensor[cls]

                        num_tokens += idx.shape[0]
                        #  ------ sanity check: if we use the org head.weight ------ # 
                        if False:
                            yyy=w[f'head_l2org.{cls}.weight']
                            zzz=w['head.weight']
                            for ii in range(len(idx)): 
                                tokenid = idx[ii]
                                if not torch.equal(yyy[:,ii], zzz[:,tokenid]): 
                                    breakpoint()  
                        # print("all good")
                        # ---------------------------- # 

                        # since we use the orig head weights, 
                        #   it's ok to concat the raw logits from multi clusters
                        logits.scatter_(dim=0, index=idx, src=x1)                    
                    
                    # breakpoint()

                    if True:
                        # all "other clusters": pseudo logits 
                        Q = sum_known_logits_exp                    
                        for i in range(0, len(CLS_OTHER)):
                            cls = CLS_OTHER[i]
                            clsprob = CLSPROBS_OTHER[i]

                            # cls: cluster id, 
                            # self.clusters[cls] list of token_ids in this cls (as scatter idx
                            idx = self.clusters_tensor[cls]
                            num_t = idx.shape[0]

                            # x1: pseudo logits over tokens inside cls, (as scatter src
                            # S_j: sum of exp logits for the cluster
                            S_j  = Q * (clsprob / CLSPROBS.sum())
                            x1 = (S_j / num_t).log() * torch.ones(num_t, device=x.device, dtype=x.dtype)
                            logits.scatter_(dim=0, index=idx, src=x1)
                        
                    # -- sanity check: we should have overwritten all prefilled 'inf' ----- #
                    assert not torch.isinf(logits).any(), "Tensor logits contains -inf values"

                    # -- sanity check: pseudo probs vs. known probs ---- #
                    #   accmulated cls probs may diff a bit ... bug or just precision issues??
                    #       (bc cls probs are sum of many token probs....)
                    if False: 
                        pseu_token_probs = torch.softmax(logits, dim=0)                    
                        for i in range(0, len(CLS)):
                            cls = CLS[i]
                            idx = self.clusters_tensor[cls]
                            cls_prob = sum(pseu_token_probs[idx])
                            print(f"{cls_prob}, {CLSPROBS[i]}")
                            # assert(cls_prob == CLSPROBS[i])
                        for i in range(0, len(CLS_OTHER)):
                            cls = CLS_OTHER[i]
                            idx = self.clusters_tensor[cls]
                            cls_prob = sum(pseu_token_probs[idx])
                            # assert(cls_prob == CLSPROBS_OTHER[i])
                        breakpoint()

                    # update statistics
                    self.stat_runs += 1
                    self.stat_loaded_cls += len(CLS)
                    self.stat_loaded_tokens += num_tokens
                    
                    # -- sanity check ---- ... expensive 
                    if False: 
                        reallogits = x @ w['head.weight']
                        # if N==200:
                        if False:
                            # if not torch.equal(reallogits,logits):
                            # XXX there might be a minor bug somewhere.... causing 
                            # minor diff in the two logitgs...
                            if not torch.allclose(reallogits,logits, rtol=0.01, atol=0.01):
                                dif=reallogits-logits
                                nzdifmask=dif!=0
                                nzidx=torch.nonzero(nzdifmask, as_tuple=False)
                                breakpoint()
                        tokens, probs= self.select_logits(reallogits, 5, 20, 0.85)
                        
                        # useful CMP: our computed logits (others filled -inf) vs. true logits
                        # if K==200, they shall equal 
                        if True:
                            print(reallogits[tokens])
                            print(logits[tokens])
                            breakpoint()

                    return logits 

                # same as value3, except that l1 project use MLP
                def _retrieve_value4(x, w):
                    # x: shape D (regardless of seq_mode
                    # l1 projection
                    # x1 = x @ w['head_l1.weight']  # shape D,K
                    # >>>>>>>>>>> CHANGED HERE <<<<<<<<<<<<<<<<<<<<<
                    x = F.relu(x @ w['head_l1fc1.weight'])
                    x1 = x @ w['head_l1fc2.weight']
                    
                    # --- select, not sampling --- # 
                    # "minK" has a high impact on speed... 
                    CLS, CLSPROBS = self.select_logits(x1, minK=3, maxK=100, minProb=.95) 
                    # CLS, CLSPROBS = self.select_logits(x1, minK=5, maxK=40, minProb=.5) 
                    # CLS, CLSPROBS = self.select_logits(x1, minK=N, maxK=N, minProb=.5)  # seems quite good? (N=200

                    #### now we've picked N cls. L2 projection.... ###
                    vocab = w['head.weight'].shape[1]   # shape D,vocab                
                    logits = torch.full((vocab,), float('-inf'), device='cuda', dtype=x1.dtype) 

                    # project x to logits
                    for i in range(0, len(CLS)):
                        cls = CLS[i]
                        clsprob = CLSPROBS[i]
                        x1 = x @ w[f'head_l2org.{cls}.weight'] 

                        # cls: cluster id, 
                        # self.clusters[cls] list of token_ids in this cl s (as scatter idx
                        # x: logits over tokens inside cls, (as scatter src
                        # idx = torch.tensor(self.clusters[cls], device='cuda')
                        idx = self.clusters_tensor[cls]

                        #  ------ sanity check: if we use the org head.weight ------ # 
                        if False:
                            yyy=w[f'head_l2org.{cls}.weight']
                            zzz=w['head.weight']
                            for ii in range(len(idx)): 
                                tokenid = idx[ii]
                                if not torch.equal(yyy[:,ii], zzz[:,tokenid]): 
                                    breakpoint()  
                        # print("all good")                        
                        # ---------------------------- # 

                        # since we use the orig head weights, 
                        #   it's ok to concat the raw logits from multi clusters
                        logits.scatter_(dim=0, index=idx, src=x1)

                    # -- sanity check ---- ... expensive 
                    if True: 
                        reallogits = x @ w['head.weight']
                        # if N==200:
                        if False:
                            # if not torch.equal(reallogits,logits):
                            # XXX there might be a minor bug somewhere.... causing 
                            # minor diff in the two logitgs...
                            if not torch.allclose(reallogits,logits, rtol=0.01, atol=0.01):
                                dif=reallogits-logits
                                nzdifmask=dif!=0
                                nzidx=torch.nonzero(nzdifmask, as_tuple=False)
                                breakpoint()
                        tokens, probs= self.select_logits(reallogits, 5, 20, 0.85)
                        
                        # useful CMP: our computed logits (others filled -inf) vs. true logits
                        # if K==200, they shall equal 
                        if True:
                            print(reallogits[tokens])
                            print(logits[tokens])
                            breakpoint()

                    return logits 
                
                if x.dim() > 1:
                    new_x = []
                    self.cached_orgx = x
                    for row in x:
                        new_x.append(self._retrieve_value3_jit(row, self.head_l1_weight, self.head_l2org_weight))
                    x = torch.stack(new_x)
                else:
                    x = self._retrieve_value3_jit(x, self.head_l1_weight, self.head_l2org_weight)

            elif w['head.weight'].dtype != torch.uint8:  # original cls head
                x = x @ w['head.weight']
            else:
                if seq_mode and full_output:
                    x = mm8_seq(x, w['head.weight'], w['head.weight_mx'], w['head.weight_rx'], w['head.weight_my'], w['head.weight_ry'])
                else:
                    x = mm8_one(x, w['head.weight'], w['head.weight_mx'], w['head.weight_rx'], w['head.weight_my'], w['head.weight_ry'])

            time_measure['cls_end'] = time.time()
            time_measure['cls_exec'] = time_measure['cls_end'] - time_measure['cls_start']

            time_measure['fwd_end'] = time.time()

            if False:    # for debugging
                print(f'fwd time: {time_measure["fwd_end"] - time_measure["fwd_start"]:.2f} sec')
                print(f'att time: {time_measure["att_exec"]:.2f} sec')
                print(f'ffn time: {time_measure["ffn_exec"]:.2f} sec')
                print(f'cls time: {time_measure["cls_exec"]:.2f} sec')

            # update global stat: 
            self.stat_time_fwd += time_measure["fwd_end"] - time_measure["fwd_start"]
            self.stat_time_att += time_measure["att_exec"]
            self.stat_time_ffn += time_measure["ffn_exec"]
            self.stat_time_cls += time_measure["cls_exec"]
            
            # breakpoint()

            return x.float(), state, layer_masks
        
    # copied from rwkv/utils.py 

    #  only sample among top items with accumulative probs > "top_p", and 
    #   ranked higher than "top_k"
    #   "size": returned sample size
    #   xzl: "replace=False" forbids same item selected multiple times
    #  return: [samples], [probs]
    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0, 
                      size=1, replace=False):
        import numpy as np
        if temperature == 0:
            temperature = 1.0
            top_p = 0
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)

        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        if probs.device.type in ['cpu', 'privateuseone']:
            probs = probs.cpu().numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0           #xzl: suppress the probs
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0    #xzl: just supress the probs
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            # xzl: here, still can choose from items with prob=0 (?
            out = np.random.choice(a=len(probs), p=probs, size=size, replace=replace)
            # return int(out)
            return out, probs[out]
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # out = torch.multinomial(probs, num_samples=size, replacement=replace)[0]  # old
            # breakpoint()
            out = torch.multinomial(probs, num_samples=size, replacement=replace)
            # TODO: complete cumulative probs
            # return int(out)
            if sum(probs[out]) < 0.3:
                print(sum(probs[out]))
                breakpoint()
            return out, probs[out]
        
    # from logits, select:
    #  at least minK, at most maxK, and stop when accmu prob > minProb
    #  return: [selected_ids], [selected_probs], [rest_ids], [rest_probs]
    def select_logits(self, logits, minK, maxK, minProb):
        import numpy as np

        probs = F.softmax(logits.float(), dim=-1)
        sorted_ids = torch.argsort(probs)
        sorted_ids = torch.flip(sorted_ids, dims=(0,)) 
        sorted_probs = probs[sorted_ids]
        # sorted_probs = torch.flip(sorted_probs, dims=(0,)) 
        # now sorted_probs in descending order
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff_idx = np.argmax(cumulative_probs >= minProb) + 1 # idx in sorted_probs
        # cutoff = float(sorted_probs[np.argmax(cumulative_probs >= minProb)])
        if cutoff_idx<minK:
            cutoff_idx=minK
        elif cutoff_idx>maxK:
            cutoff_idx=maxK
        return sorted_ids[:cutoff_idx], sorted_probs[:cutoff_idx], \
            sorted_ids[cutoff_idx:], sorted_probs[cutoff_idx:],

    # same as above, but **cpu jit friendly**
    @MyFunction
    def select_logits_jit(self, logits, minK: int, maxK: int, minProb: float):
        probs = F.softmax(logits.float(), dim=-1)
        sorted_ids = torch.argsort(probs)
        sorted_ids = torch.flip(sorted_ids, dims=(0,)) 
        sorted_probs = probs[sorted_ids]
        # sorted_probs = torch.flip(sorted_probs, dims=(0,)) 
        # now sorted_probs in descending order

        # below: works unless using torch script (.numpy() unsupported)
        '''
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff_idx = np.argmax(cumulative_probs >= minProb) + 1 # idx in sorted_probs        
        ### cutoff = float(sorted_probs[np.argmax(cumulative_probs >= minProb)])
        '''        
        ###### torchscript (cpu) friendly version....
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # cutoff_idx = torch.argmax(cumulative_probs >= minProb).item() + 1  # idx in sorted_probs
        condition = cumulative_probs >= minProb
        indices = torch.nonzero(condition)
        cutoff_idx = indices[0].item() + 1 if indices.numel() > 0 else 0  # idx in sorted_probs

        if cutoff_idx<minK:
            cutoff_idx=minK
        elif cutoff_idx>maxK:
            cutoff_idx=maxK
        return sorted_ids[:cutoff_idx], sorted_probs[:cutoff_idx], \
            sorted_ids[cutoff_idx:], sorted_probs[cutoff_idx:],
