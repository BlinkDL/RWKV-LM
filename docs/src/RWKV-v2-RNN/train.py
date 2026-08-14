########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

### Step 1: set training data ##########################################################################

datafile = "enwik8"
datafile_encoding = 'utf-8'
# datafile_encoding = 'utf-16le'

### Step 2: set model size #############################################################################

ctx_len = 1024        # ===> increase T_MAX in model.py if your ctx_len > 1024
n_layer = 6
n_embd = 512

# 'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
model_type = 'RWKV'

### Step 3: set batch size #############################################################################

# ===> batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in model.py
# For example, if your batch_size = 20, you can set B_GROUP_FORWARD = 4, B_GROUP_BACKWARD = 2
# If you see "CUDA out of memory", reduce it. Use GPU-Z to find the highest value for your VRAM.
batch_size = 12

### Step 4: set learning rate, training mini-epochs #######################################################

lr_init = 6e-4
lr_final = 1e-5
# the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
n_epoch = 500
# 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_frequency = 30
epoch_save_path = 'trained-'

epoch_length_fixed = 10000

########################################################################################################

# import src.utils
# src.utils.set_seed(42) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

grad_norm_clip = 1.0
warmup_tokens = 0

betas = (0.9, 0.99)
eps = 4e-9

num_workers = 0

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile)
train_dataset = Dataset(open(
    datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

    # # # load a trained model. remember to change random seed
    # m2 = torch.load('trained-61.pth')
    # model.load_state_dict(m2)

    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
