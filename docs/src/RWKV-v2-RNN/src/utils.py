########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data, ctx_len, epoch_length_fixed):
        print('building token list...', end=' ')
        unique = sorted(list(set(data)))
        # print()
        # for u in unique:
        #     print(u, end=' ')
        # print('\n\n')

        xx = 0
        xxObj = {}
        for u in unique:
            xxObj[xx] = u
            xx += 1
        with open('vocab.json', "w", encoding="utf-16") as vocab_file:
            vocab_file.write(json.dumps(xxObj, ensure_ascii=False))

        data_size, vocab_size = len(data), len(unique)
        print('data has %d tokens, %d unique.' % (data_size, vocab_size))
        self.stoi = {ch: i for i, ch in enumerate(unique)}
        self.itos = {i: ch for i, ch in enumerate(unique)}
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return self.epoch_length_fixed

    def __getitem__(self, idx):
        # cheat: pick a random spot in dataset
        i = np.random.randint(0, len(self.data) - (self.ctx_len + 1))
        chunk = self.data[i:i+self.ctx_len+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long,
                         device=torch.device('cuda'))
        y = torch.tensor(dix[1:], dtype=torch.long,
                         device=torch.device('cuda'))
        return x, y


class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
            self.word_table = json.load(result_file)

        self.vocab_size = len(self.word_table)

        self.stoi = {v: int(k) for k, v in self.word_table.items()}
        self.itos = {int(k): v for k, v in self.word_table.items()}

        self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'

        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        # out[self.UNKNOWN_CHAR] = -float('Inf')

        lastChar = int(x[-1])

        probs = F.softmax(torch.tensor(out), dim=-1)

        if self.itos[lastChar] == '\n':
            top_p = top_p_newline
        else:
            top_p = top_p_usual

        sorted_probs, s_index = torch.sort(probs, descending=True)

        # for j in range(30):
        #     pp = sorted_probs[j].item()
        #     if pp < 0.005:
        #         break
        #     ss = self.itos[int(s_index[j])].replace('\n','_')
        #     print(f'{math.floor(pp*100):>3.0f}{ss}', end='')
        # print('')

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

        probs[probs < cutoff] = 0
        # print("[" + str(round(cutoff,4)) + ' ' + str(round(to_float(sum(probs)),3)) + "]", end = "")

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return torch.multinomial(probs, num_samples=1)[0]


def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
