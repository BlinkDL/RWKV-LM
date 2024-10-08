'''
test rwkv inference engine
cf: https://pypi.org/project/rwkv/

'''
import sys, os
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from datasets import load_dataset

# run chat app on the inference engine (rwkv)
# collect sparsity data for MLP training

os.environ["RWKV_JIT_ON"] = '1'

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

RWKV_HOME = os.environ.get("RWKV_HOME") # User specific. See env-amd.sh
model_path=f'{RWKV_HOME}/RWKV-v5/out/04b-pre-x59-8x-sparsity/rwkv-2405'
sparse_path=f'{RWKV_HOME}/RWKV-v5/out/04b-pre-x59-8x-sparsity'

COLLECT_SPARSITY_DATA = True
print(f'Loading model - {model_path}')

def my_print(s):
    print(s, end='', flush=True)

# xzl: for strategy, cf: https://pypi.org/project/rwkv/ for more ex
#
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32


model = RWKV(model=model_path, 
             strategy='cuda fp16', 
             verbose=True,
             sparse_outpath=sparse_path,
             )
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
N_LAYERS = model.args.n_layer

dict_tensors = {}

for n in range(N_LAYERS):
    dict_tensors[n] = []

data = load_dataset("Open-Orca/OpenOrca", data_files="1M-GPT4-Augmented.parquet")["train"]

ctx_list = []
for row in data:
    attr1 = row["system_prompt"]
    attr2 = row["question"]
    ctx_list.append( f"\n{attr1}\n{attr2}")

print(f"len of ctx_list: {len(ctx_list)}")


args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

"""
This generates the data for the sparsity training
""" 

p_tensors = []
for i, ctx in enumerate(ctx_list[:500]):
    print(f"\nPrompt [{i}]: ...")
    print(ctx, end='')
    _, t_data = pipeline.generate(ctx, token_count=200, args=args, callback=my_print,
                                  collect_sparse_data=COLLECT_SPARSITY_DATA)
    p_tensors.append(t_data)

# len(p_tensors) == # of prompts
for p in p_tensors:
    # len(t) == # of tokens generated == 199 except for seq
    for t in p:
        for n in range(N_LAYERS):
            dict_tensors[n].append(t[n])

# --- sanity check
for n in range(N_LAYERS):
    print(f"layer {n}: {len(dict_tensors[n])}")
    outpath_query=f'{model_dir}/FFN.key-layer{n}-query.npy'
    try:
        tensor_list = torch.load(outpath_query, weights_only=True)
        print(f"\n{len(tensor_list)}")
        if not isinstance(tensor_list, list):
            tensor_list = [tensor_list]
    except FileNotFoundError:
        tensor_list = []

    tensor_list += dict_tensors[n]
    print(f"{len(tensor_list)}")
    #torch.save(tensor_list, outpath_query)
