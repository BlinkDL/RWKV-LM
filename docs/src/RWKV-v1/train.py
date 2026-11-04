########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, time, math, random, json, datetime, logging
import numpy as np
import torch
from torch.utils.data import Dataset
from src.trainer import Trainer, TrainerConfig
from src.model import GPT, GPTConfig
from src.utils import set_seed

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

# RWKV       : our new model - fastest when ctx_len is long - VRAM friendly - good performance
# MHA_rotary : usual MultiheadAttention+Rotary+GeGLU - not as good
# MHA_shift  : with time-shift - good performance
# MHA_pro    : slow (lots of tricks) - VRAM hungry - very good performance
model_type = 'RWKV'

# datafile = u"V:\\NLP\\text8"
# datafile = u"V:\\NLP\\enwik8"
datafile = u"V:\\NLP\\simplebooks\\simplebooks-92-raw\\train.txt"
datafile_encoding = 'utf-8'
# datafile = u"D:\\NLP-Data\\ww100M.txt"
# datafile = u"D:\\NLP-Data\\__2019.txt"
# datafile = u"Y:\\BlinkNLP\\_txt_\\txt\\_all.txt"
# datafile = u"V:\\NLP\\enwik8-shift-300.bpe"
# datafile_encoding = 'utf-16'
# datafile = u"V:\\NLP\\simplebooks-shift-utf32.word"
# datafile_encoding = 'utf-32'

datafile_type = 0 # use 0 for char-level english. use 1 for chinese. only affects some RWKV hyperparametrs 

#################################### VERY IMPORTANT ####################################
epoch_save_frequency = 10                            # 0 = never, 1 = every 'epoch', 2 = every two 'epoch', etc.
epoch_save_path = 'trained-'

batch_size = 32                                      # if you see "CUDA out of memory", reduce this.
                                                     # if you have good GPU, increase this.
                                                     # use GPU-Z to find the highest value for your VRAM.

n_epoch = 100                                        # the 'epoch' here is actually very short (and of fixed length)
########################################################################################

model_level = 'character' # 'character' (recommended) or 'word'

ctx_len = 256 # context length, try 512 or 1024 if you have good GPU
n_layer = 6   # try 12 for 100M, 24 for 300M
n_head = 8    # try 12 for 100M, 16 for 300M

n_embd = n_head * 64
n_attn = n_embd
n_ffn = n_embd

lr_init = 6e-4 if model_type == 'RWKV' else 4e-4    # RWKV can use higher lr.  8e-4 = 0.0008   4e-4 = 0.0004
lr_final = 4e-5

betas = (0.9, 0.99) if model_type == 'RWKV' else (0.9, 0.99)
eps = 4e-9
weight_decay = 0 if model_type == 'RWKV' else 0.01  # wd is not useful when we have enough data

epoch_length_fixed = 10000                          # make an 'epoch' very short, so we can see the training progress

######## special hyperparameters for RWKV model ########
rwkv_emb_scale = 0.4                                # scale of initial embedding. 0.4 is a good choice
rwkv_tiny_attn = 0#64 if (datafile_type == 0 and ctx_len > 600) else 0 # extra tiny attention dim, useful for long ctx char-level english
rwkv_tiny_head = 1                                  # 1 is good enough. 8 is slow
# n_side_proj = 512                                 # extra 'side projection', quite useful for BPE models 

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile)

class Dataset(Dataset):
    def __init__(self, data, model_level, ctx_len):
        print('building token list...', end=' ')
        if model_level == 'word':
            import re
            data = re.sub(r'(\n|\.|\,|\?|\!|\:|\;|\-|\—|\||\'|\"|\`|\(|\)|[0-9]|\[|\]|\{|\}|\=|\+|\*|\\|\/|\~|\&|\$|\#|\%)', r' \g<0> ', data)
            data = re.sub(' +',' ',data)
            print('splitting token...')
            data = data.lower().split(' ')
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
        print('data has %d %ss, %d unique.' % (data_size, model_level, vocab_size))
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
                rwkv_emb_scale=rwkv_emb_scale, rwkv_tiny_attn=rwkv_tiny_attn, rwkv_tiny_head=rwkv_tiny_head,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_attn=n_attn, n_ffn=n_ffn))

# load a trained model
# model.load_state_dict(torch.load('trained-xxx.pth').state_dict())

print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas', betas, 'eps', eps, 'wd', weight_decay, 'ctx', ctx_len, 'layer', n_layer, 'head', n_head, 'embd', n_embd, 'attn', n_attn, 'ffn', n_ffn)
tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size, weight_decay=weight_decay,
                        learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps,
                        warmup_tokens=0, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=0, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()

torch.save(model, 'trained-' + trainer.get_run_name() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
