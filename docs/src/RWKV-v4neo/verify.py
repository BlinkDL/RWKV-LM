########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# this is for verifying the results of different models and make sure they agree with each other

import os, sys, types
import numpy as np
import torch
np.set_printoptions(precision=4, suppress=True, linewidth=200)
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

os.environ['RWKV_FLOAT_MODE'] = 'bf16' # bf16 or fp32
os.environ['RWKV_RUN_DEVICE'] = 'cuda' # currently model_train requires CUDA
RUN_DEVICE = os.environ['RWKV_RUN_DEVICE']

TOKEN_MODE = 'pile'

if TOKEN_MODE == 'pile':
    WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
    MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221003-6783'
    n_layer = 32
    n_embd = 2560
    ctx_len = 1024
    UNKNOWN_CHAR = None

from src.utils import TOKENIZER
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == 'pile':
    tokenizer.vocab_size = 50277

########################################################################################################

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_T_MAX"] = str(ctx_len)

from src.model_run import RWKV_RNN
from src.model import RWKV

args = types.SimpleNamespace()
args.vocab_size = tokenizer.vocab_size
args.ctx_len = ctx_len
args.n_embd = n_embd
args.n_layer = n_layer
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
model_train = RWKV(args).to(RUN_DEVICE)

if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
    model_train = model_train.half()
elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
    model_train = model_train.bfloat16()

print('loading ' + MODEL_NAME)
m2 = torch.load(MODEL_NAME + '.pth', map_location='cpu')
model_train.load_state_dict(m2)

if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
    model_train = model_train.half()
elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
    model_train = model_train.bfloat16()

args.MODEL_NAME = MODEL_NAME
args.RUN_DEVICE = RUN_DEVICE
args.FLOAT_MODE = os.environ['RWKV_FLOAT_MODE']
model_rnn = RWKV_RNN(args)

########################################################################################################

print(f"\nVerifying {os.environ['RWKV_RUN_DEVICE']} {os.environ['RWKV_FLOAT_MODE']}")

# context = '\nIn a'
context = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'

if TOKEN_MODE == 'pile':
    ctx = tokenizer.tokenizer.encode(context)
print(f'input len {len(ctx)} data {ctx}')

########################################################################################################

with torch.no_grad():
    print('\nRWKV-train output')
    out = model_train.forward(torch.tensor([ctx]).to(RUN_DEVICE))[0].detach().cpu().float().numpy()
    print(out, '\n')

    print('\nRWKV-RNN output')
    state = None
    out = None
    src_len = len(ctx)
    for i in range(src_len):
        x = ctx[:i+1]
        out, state = model_rnn.forward(x, state)
        if i < 3 or i >= src_len - 3:
            print(out.detach().cpu().numpy())
        if i == 2:
            print('...')
