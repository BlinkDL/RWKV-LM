########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# this is for verifying the results of different models and make sure they agree with each other

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RUN_DEVICE = 'cuda'

import torch
from src.model_run import RWKV_RNN, RWKV_GPT
from src.model import GPT, GPTConfig

ctx_len = 1024
n_layer = 6
n_embd = 512
model_type = 'RWKV'

model_name = 'trained-1'

from src.utils import TOKENIZER
tokenizer = TOKENIZER('vocab', UNKNOWN_CHAR=' ')

########################################################################################################

model_train = GPT(GPTConfig(tokenizer.vocab_size, ctx_len, model_type=model_type, n_layer=n_layer, n_embd=n_embd)).cuda().half()
print('loading ' + model_name)
m2 = torch.load(model_name + '.pth', map_location=RUN_DEVICE)
model_train.load_state_dict(m2)

model_rnn = RWKV_RNN(model_name, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len)
model_gpt = RWKV_GPT(model_name, RUN_DEVICE, model_type, tokenizer.vocab_size, n_layer, n_embd, ctx_len).cuda()

########################################################################################################

context = '\nIn a'
ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
print(f'input len {len(ctx)} data {ctx}')

########################################################################################################

print('\nRWKV-GPT output')
out = model_gpt.forward(torch.tensor(ctx).unsqueeze(0).cuda())[0].detach().cpu().numpy()
print(out)

print('\nRWKV-RNN output')
model_rnn.clear()
src_len = len(ctx)
for i in range(src_len):
    x = ctx[:i+1]
    out = model_rnn.run(x)
    if i < 3 or i >= src_len - 3:
        print(torch.tensor(out).detach().cpu().numpy())
    if i == 2:
        print('...')

print('\nRWKV-train output')
out = model_train.forward(torch.tensor([ctx]).cuda())[0][0].detach().cpu().numpy()
print(out, '\n')
