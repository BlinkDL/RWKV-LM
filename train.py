########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, time, math, random, json, datetime
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from src.trainer import Trainer, TrainerConfig
from src.model import GPT, GPTConfig
from src.utils import set_seed

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)

# RWKV       : our new model - fastest when ctx_len is long - VRAM friendly - good performance
# MHA_rotary : usual Multi-head Attention+Rotary+GeGLU - not as good
# MHA_shift  : with time-shift - good performance
# MHA_pro    : slow (lots of tricks) - VRAM hungry - very good performance
model_type = 'RWKV'

# datafile = u"V:\\NLP\\text8"
# datafile = u"V:\\NLP\\enwik8"
datafile = u"V:\\NLP\\simplebooks\\simplebooks-92-raw\\train.txt"
datafile_encoding = 'utf-8'
# datafile = u"Y:\\BlinkNLP\\_txt_\\txt\\_all.txt"
# datafile_encoding = 'utf-16'

model_level = 'character' # 'character' or 'word'

ctx_len = 256                                       # context length
n_layer = 5
n_head = 8
n_embd = n_head * 64

batch_size = 64

n_epoch = 50                                        # the 'epoch' here is actually very short (and of fixed length)
lr_init = 6e-4 if model_type == 'RWKV' else 4e-4    # seems RWKV can use higher lr
lr_final = 2e-4

betas = (0.9, 0.99)
weight_decay = 0 if model_type == 'RWKV' else 0.01  # seems wd is not very useful when we have enough data
epoch_length_fixed = 10000                          # make an 'epoch' very short, so we can see the training progress

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile)

class Dataset(Dataset):
    def __init__(self, data, model_level, ctx_len):
        print('building token list...')
        if model_level == 'word':
            import re
            data = re.sub(r'(\n|\.|\,|\?|\!|\:|\;|\-|\—|\||\'|\"|\`|\(|\)|[0-9]|\[|\]|\{|\}|\=|\+|\*|\\|\/|\~|\&|\$|\#|\%)', r' \g<0> ', data)
            data = re.sub(' +',' ',data)
            print('splitting token...')
            data = data.lower().split(' ')
        unique = sorted(list(set(data)))
        # for u in unique:
        #     print(u, end=' ')
        data_size, vocab_size = len(data), len(unique)
        print('\n\ndata has %d %ss, %d unique.' % (data_size, model_level, vocab_size))
        self.stoi = { ch:i for i,ch in enumerate(unique) }
        self.itos = { i:ch for i,ch in enumerate(unique) }
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return epoch_length_fixed

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data) - (self.ctx_len + 1)) # cheat: pick a random spot in dataset
        chunk = self.data[i:i+self.ctx_len+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

train_dataset = Dataset(open(datafile, "r", encoding=datafile_encoding).read(), model_level, ctx_len)

########################################################################################################
# Train model
########################################################################################################

model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd))

print('model', model_type, 'total epoch', n_epoch, 'batch_size', batch_size, 'n_layer', n_layer, 'n_head', n_head, 'n_embd', n_embd, 'len', ctx_len)
tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size, weight_decay=weight_decay,
                        learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas,
                        warmup_tokens=0, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=0)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()

torch.save(model, 'trained-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')

########################################################################################################
# Run model to generate text
########################################################################################################

from src.utils import sample_logits

NUM_OF_RUNS = 5
LENGTH_OF_EACH = 300

for run in range(NUM_OF_RUNS):
    context = "it was"

    if model_level == 'word':
        x = np.array([train_dataset.stoi[s] for s in context.strip().lower().split(' ')], dtype=np.int64)
    else:
        x = np.array([train_dataset.stoi[s] for s in context], dtype=np.int64)

    real_len = len(x)
    if real_len < ctx_len:
        x = np.pad(x, (0, ctx_len - real_len))
    print_begin = 0
        
    for i in range(LENGTH_OF_EACH):

        if i == 0:
            print(('-' * 80) + '\n' + context, end = '')
            print_begin = real_len

        with torch.no_grad():
            xxx = torch.tensor(x[-ctx_len:], dtype=torch.long)[None,...].to("cuda:0")
            out, _ = model(xxx)
        pos = -1 if real_len >= ctx_len else real_len - 1

        char = sample_logits(out, pos, temperature=1.0, min_p_pow=2.0, min_p_ratio=0.02) # our special sampling method
    
        if real_len < ctx_len:
            x[real_len] = char
        else:
            x = np.append(x, char)
        real_len += 1

        if i % 10 == 9 or i == LENGTH_OF_EACH-1:
            if model_level == 'word':
                completion = ' ' + ' '.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
                completion = completion.replace('\n ', '\n')
            else:
                completion = ''.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
            print(completion, end = '')
            print_begin = real_len
    print()
