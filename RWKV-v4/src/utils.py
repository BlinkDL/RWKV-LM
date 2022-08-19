########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
try:
    NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
except:
    NUM_GPUS = 1

import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data, ctx_len, epoch_length_fixed):
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed
        self.data = data

        if 'MMapIndexedDataset' in str(type(self.data)):
            self.vocab_size = int(os.environ['VOCAB_SIZE'])
            print('current vocab size =', self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data._bin_buffer) // 2
            print(f'data has {self.data_size} tokens.')
        elif 'numpy' in str(type(self.data)):
            self.vocab_size = int(os.environ['VOCAB_SIZE'])
            print('current vocab size =', self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            print(f'data has {self.data_size} tokens.')
        else:
            print('building token list...', end=' ')
            unique = sorted(list(set(data)))
            self.vocab_size = len(unique)
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
            self.data_size = len(self.data)
            print('data has %d tokens, %d unique.' % (self.data_size, self.vocab_size))
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.epoch_length_fixed // NUM_GPUS

    def __getitem__(self, idx):
        #
        # we are cheating: pick a random spot in dataset
        #
        i = np.random.randint(0, self.data_size - (self.ctx_len + 1))
        if 'MMapIndexedDataset' in str(type(self.data)):
            dix = self.data.get(idx=0, offset=i, length=self.ctx_len + 1).astype(int)
        elif 'numpy' in str(type(self.data)):
            dix = self.data[i:i+self.ctx_len+1]
        else:
            dix = [self.stoi[s] for s in self.data[i:i+self.ctx_len+1]]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] == WORD_NAME[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
        else:
            self.charMode = True
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

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
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
