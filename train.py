import os, sys, time, math, random, json, datetime
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from src.trainer import Trainer, TrainerConfig
from src.model import GPT, GPTConfig
from src.utils import set_seed

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)

model_type = 'RWKV' # 'RWKV' or 'RotaryMHA'

datafile = u"V:\\NLP\\simplebooks\\simplebooks-92-raw\\train.txt" # https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip
model_level = 'character' # 'character' or 'word'

ctx_size = 256 if 'character' else 128
nLayers = 5
nHead = 8
nEmb = 512

nepoch = 50
nbatchsz = 64
epoch_length_fixed = 10000 # make an epoch very short, so we can see the training progress

########################################################################################################

print("loading data...", end="")

class Dataset(Dataset):
    def __init__(self, data, model_level, ctx_size):
        if model_level == 'word':
            data = data.replace('\n', ' \n ').replace('  ', ' ').split(' ')

        unique = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(unique)
        self.stoi = { ch:i for i,ch in enumerate(unique) }
        self.itos = { i:ch for i,ch in enumerate(unique) }
        print('data has %d %ss, %d unique.' % (data_size, model_level, vocab_size))
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return epoch_length_fixed

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data) - (self.ctx_size + 1)) # CHEAT: pick a spot in the dataset at random
        chunk = self.data[i:i+self.ctx_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

train_dataset = Dataset(open(datafile, "r", encoding="utf-8").read(), model_level, ctx_size)

########################################################################################################

model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_size, model_type=model_type,
                n_layer=nLayers, n_head=nHead, n_embd=nEmb))

print('model', model_type, 'total epoch', nepoch, 'batchsz', nbatchsz, 'nLayers', nLayers, 'nHead', nHead, 'nEmb', nEmb, 'len', ctx_size)
tconf = TrainerConfig(model_type=model_type, max_epochs=nepoch, batch_size=nbatchsz,
                        learning_rate=6e-4 if model_type == 'RWKV' else 4e-4, betas=(0.9, 0.99), # RWKV can use higher LR
                        lr_decay=True, lr_final=2e-4, warmup_tokens=0, final_tokens=nepoch*len(train_dataset)*ctx_size, num_workers=0)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()

torch.save(model, 'trained-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')

########################################################################################################

from src.utils import sample_logits

MAX_LEN = ctx_size
NUM_OF_RUNS = 5
LENGTH_OF_EACH = 300

for run in range(NUM_OF_RUNS):
    context = "It was"

    x = np.array([train_dataset.stoi[s] for s in context], dtype=np.int64)

    real_len = len(x)
    if real_len < MAX_LEN:
        x = np.pad(x, (0, MAX_LEN - real_len))
    print_begin = 0
        
    for i in range(LENGTH_OF_EACH):

        if i == 0:
            print(('-' * 80) + '\n' + context, end = '')
            print_begin = real_len

        with torch.no_grad():
            xxx = torch.tensor(x[-MAX_LEN:], dtype=torch.long)[None,...].to("cuda:0")
            out, _ = model(xxx)
        pos = -1 if real_len >= MAX_LEN else real_len - 1

        char = sample_logits(out, pos, temperature=1.0, min_p_pow=2.0, min_p_ratio=0.02)
    
        if real_len < MAX_LEN:
            x[real_len] = char
        else:
            x = np.append(x, char)
        real_len += 1

        if i % 10 == 9 or i == LENGTH_OF_EACH-1:
            completion = ''.join([train_dataset.itos[int(i)] for i in x[print_begin:real_len]])
            print(completion, end = '')
            print_begin = real_len
    print()
