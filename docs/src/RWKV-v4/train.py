########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import logging, types
from src.utils import Dataset
import torch
import numpy as np
from src.binidx import MMapIndexedDataset

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

# if False: # True False ---> Set to False if you don't understand it
#     print("\n\n[[[ SPECIAL DEBUG MODE FOR MYSELF. DON'T ENABLE THIS IF YOU DON'T UNDERSTAND IT ]]]\n\n")
#     import src.utils
#     src.utils.set_seed(42) # make training deterministic (including dataloader). if you are doing this, remember to change seed when you load a model (otherwise the dataloader loads old samples)

########################################################################################################
# Step 1: set training data & cfg
########################################################################################################

EXPRESS_PILE_MODE = False # True: express mode for fine-tuning a pile model // False: usual training

EXPRESS_PILE_MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
EXPRESS_PILE_MODEL_TYPE = 'RWKV-4-Pile-169M'
# EXPRESS_PILE_MODEL_NAME = 'RWKV-4-Pile-430M-20220808-8066'
# EXPRESS_PILE_MODEL_TYPE = 'RWKV-4-Pile-430M'
# EXPRESS_PILE_MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
# EXPRESS_PILE_MODEL_TYPE = 'RWKV-4-Pile-1B5'

########################################################################################################

datafile = "../data/enwik8" # your data
datafile_encoding = 'utf-8' # 'utf-8' / 'utf-16le' / 'numpy' (for fine-tuning pile models) / 'binidx' (the Megatron-LM 'binidx' format)

# datafile = 'my-gpt_seq_document'
# datafile_encoding = 'binidx'

if EXPRESS_PILE_MODE:
    datafile = 'train.npy' # use 'prepare-data.py' in https://github.com/BlinkDL/RWKV-v2-RNN-Pile/tree/main/RWKV-v3 to tokenize .txt into .npy
    datafile_encoding = 'numpy'

#
# set VOCAB_SIZE = 0 (auto-compute) if you are training a char-level LM from scratch
# set VOCAB_SIZE = 50277 for fine-tuning pile models
# set VOCAB_SIZE = your_vocab_size for 'binidx' data
#
os.environ['VOCAB_SIZE'] = '0'
if EXPRESS_PILE_MODE:
    os.environ['VOCAB_SIZE'] = '50277'

#
# Currently it's slow to initialize a new model. Hence I suggest this procedure for multi-GPU training:
# 1) set RWKV_NUM_GPUS = '1' and let it run for 1 miniEpoch and it will save a trained-1.pth
# 2) set RWKV_NUM_GPUS = '8' (or your #GPU), batch_size = single_gpu_batchsz * RWKV_NUM_GPUS,
#    EPOCH_BEGIN = 1, LOAD_MODEL = True, and it will load 'trained-1.pth' and continue the training from it
#
os.environ['RWKV_NUM_GPUS'] = '1' # num of GPUs to use

#
# 'bf16' (fast & stable)
# 'fp16' (fast & will overflow after training a large model for very long. can be solved in the future)
# 'tf32' (decent speed & stable)
# 'fp32' (!!!very slow!!! only for verification)
os.environ['RWKV_FLOAT_MODE'] = 'bf16'

os.environ['RWKV_DEEPSPEED'] = '1' # Use DeepSpeed? 0 = False, 1 = True

if int(os.environ['RWKV_NUM_GPUS']) == 1: # Usually you don't need DeepSpeed for 1 GPU training.
    os.environ['RWKV_DEEPSPEED'] = '0'    # However, sometimes DeepSpeed saves VRAM even for 1 GPU training. So you shall try it.

os.environ['USE_WANDB'] = '0' # wandb logging. 0 = False, 1 = True

########################################################################################################
# Step 2: set model details
########################################################################################################

EPOCH_BEGIN = 0 # begins with miniEpoch = EPOCH_BEGIN
LOAD_MODEL = False # shall we load the #EPOCH_BEGIN model and continue the training from it?

n_layer = 6
n_embd = 512
ctx_len = 1024 # increase T_MAX in src/model.py if your ctx_len is longer

model_type = 'RWKV' # 'RWKV' or 'RWKV-ffnPre' (sometimes better)

# there is also a RWKV_HEAD_QK_DIM in model.py and model_run.py
# set it to 256, then it's using my headQK trick (a tiny attention) to improve loss
# set it to 0, then it's a pure RNN (attention-free)

if EXPRESS_PILE_MODE:
    LOAD_MODEL = True
    if EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-169M':
        n_layer = 12
        n_embd = 768
        ctx_len = 1024
    elif EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-430M':
        n_layer = 24
        n_embd = 1024
        ctx_len = 1024
    elif EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-1B5':
        n_layer = 24
        n_embd = 2048
        ctx_len = 1024

########################################################################################################
# Step 3: set batch size & learning rate etc.
########################################################################################################

# if you see "CUDA out of memory", reduce batch_size. Use nvidia-smi to find the highest value for your GPU.
batch_size = 12 * int(os.environ['RWKV_NUM_GPUS'])
assert (batch_size % int(os.environ['RWKV_NUM_GPUS']) == 0)

# By default we are using exponential LR decay.
# Here are my suggestions for training.
# Let's say you are training a L6-D512 model.
# 1) Set lr_init = lr_final = 8e-4. Let it run for some mini-epochs, until you feel like reducing LR.
# 2) Check epoch_save_frequency and make sure the partially-trained model is saved. Ctrl+C to stop the run.
# 3) Set lr_init = 8e-4, lr_final = 1e-5, betas = (0.9, 0.999).
# 4) Set EPOCH_BEGIN & LOAD_MODEL to load the partially-trained model. Continue the training.
# 
# For L12-D768, set lr_init = 6e-4. For L24-D1024, set lr_init = 4e-4. For L24-D2048, set lr_init = 3e-4.

lr_init = 8e-4
lr_final = 1e-5

# the mini-epoch is very short and of fixed length (length = ctx_len * epoch_length_fixed tokens)
n_epoch = 500
epoch_length_fixed = (10000 // batch_size) * batch_size # feel free to increase it if you have lots of GPU

# epoch_save_frequency 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, ...
epoch_save_frequency = 10
epoch_save_path = 'trained-'

if EXPRESS_PILE_MODE:
    if EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-169M':
        lr_init = 2e-5
    else:
        lr_init = 1e-5
    lr_final = 1e-5
    n_epoch = 100000

### misc stuffs ########################################################################################

if LOAD_MODEL and EPOCH_BEGIN > 0: # we are not saving gradients, so let's have some warmup if we load a model
    warmup_tokens = 50 * ctx_len * batch_size // NUM_GPUS
else:
    warmup_tokens = 0

betas = (0.9, 0.99) # set betas = (0.9, 0.999) if your model has been trained for a while
eps = 1e-8

num_workers = 1 # DataLoader worker. I only tested num_workers = 1

NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
os.environ['RWKV_LOAD_MODEL'] = str(LOAD_MODEL)
MODEL_NAME = epoch_save_path + str(EPOCH_BEGIN)

if EXPRESS_PILE_MODE:
    betas = (0.9, 0.999)
    MODEL_NAME = EXPRESS_PILE_MODEL_NAME

torch.backends.cudnn.benchmark = True
if os.environ['RWKV_FLOAT_MODE'] == 'fp32':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

########################################################################################################
# Load data
########################################################################################################

print(f'loading {datafile_encoding} data... ' + datafile)
if datafile_encoding == 'binidx':
    train_dataset = Dataset(MMapIndexedDataset(datafile), ctx_len, epoch_length_fixed)
elif datafile_encoding == 'numpy':
    train_dataset = Dataset(np.load(datafile).astype('int'), ctx_len, epoch_length_fixed)
else:
    train_dataset = Dataset(open(datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

########################################################################################################
# Train model
########################################################################################################

if __name__ == '__main__':
    from src.trainer import Trainer, TrainerConfig

    print('\nmodel', model_type, os.environ['RWKV_FLOAT_MODE'], 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, '\n')

    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    m_cfg = types.SimpleNamespace()
    m_cfg.model_type = model_type
    m_cfg.n_layer = n_layer
    m_cfg.n_embd = n_embd
    m_cfg.EPOCH_BEGIN = EPOCH_BEGIN
    m_cfg.LOAD_MODEL = LOAD_MODEL
    m_cfg.MODEL_NAME = MODEL_NAME

    if os.environ['RWKV_DEEPSPEED'] == '0':
        if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=16)            
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision='bf16')
        elif '32' in os.environ['RWKV_FLOAT_MODE']:
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=32)
    else:
        from pytorch_lightning.strategies import DeepSpeedStrategy
        
        DEEPSPEED_CFG = {
            "zero_allow_untested_optimizer":True,
            "zero_optimization":{
                "stage":2,
                "contiguous_gradients":True,
                "overlap_comm":True,
                "allgather_partitions":True,
                "reduce_scatter":True,
                "allgather_bucket_size":200000000,
                "reduce_bucket_size":200000000,
                "sub_group_size":1000000000000
            },
            "activation_checkpointing":{
                "partition_activations":False,
                "cpu_checkpointing":False,
                "contiguous_memory_optimization":False,
                "synchronize_checkpoint_boundary":False
            },
            "aio":{
                "block_size":1048576,
                "queue_depth":8,
                "single_submit":False,
                "overlap_events":True,
                "thread_count":1
            },
            "gradient_clipping": 1.0,
            "gradient_accumulation_steps": 1,
        }
        if NUM_GPUS == 1:
            DEEPSPEED_CFG['zero_optimization'] = {
                "stage":1, # saves some VRAM
                "contiguous_gradients":False,
                "overlap_comm":False,
                "allgather_partitions":False,
                "reduce_scatter":False,
                "allgather_bucket_size":200000000,
                "reduce_bucket_size":200000000,
                "sub_group_size":1000000000000
            }

        if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            DEEPSPEED_CFG["fp16"] = {
                "fp16": True,
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 12,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision=16)
            
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            DEEPSPEED_CFG["bf16"] = {
                "enabled": True
            }
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision='bf16')

        elif '32' in os.environ['RWKV_FLOAT_MODE']:
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision=32)

        print(trainer._strategy.config)
    
    trainer.run(m_cfg, train_dataset, None, tconf)
