########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from genericpath import exists
from typing import List
from src.model_run import RWKV_RNN
import numpy as np
import math
import os
import sys
import types
import time
import gc
import torch
from src.utils import TOKENIZER
from tqdm import tqdm
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import inquirer
files = os.listdir()
# filter by ending in .pth
files = [f for f in files if f.endswith(".pth")]

questions = [
    inquirer.List('file',
                  message="What model do you want to use?",
                  choices=files,
                  ),
]
file = inquirer.prompt(questions)["file"]

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = {}
argsnums = {}

########################################################################################################
# Step 1: set model & config
# Do this first: pip install torchdynamo
########################################################################################################


TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
vocab_size = 50277


# 'cpu' (already very fast) // 'cuda' // proc (faster then cpu, uses a fraction of the vram of cuda)
args["RUN_DEVICE"] = inquirer.prompt([inquirer.List('RUN_DEVICE',
                                                    message="What device do you want to use?",
                                                    choices=[
                                                        "cpu", "cuda"],
                                                    )])["RUN_DEVICE"]


# how many layers to offload to cuda, smaller number is slower, but uses less vram. // 0 -> n_layer // use to speed up proc as well
if (args["RUN_DEVICE"] == "cuda"):
    argsnums["cudalayers"] = inquirer.text(
        message="How many layers to offload to cuda? (default:all)")
    if argsnums["cudalayers"] == "":
        argsnums["cudalayers"] = 100
    else:
        argsnums["cudalayers"] = int(argsnums["cudalayers"])
# fp32 // bf16 (saves VRAM, slightly less accurate) // fp16 (saves VRAM, slightly less accurate, can only be used with cuda, sometimes faster)
args["FLOAT_MODE"] = inquirer.prompt([inquirer.List('FLOAT_MODE',
                                                    message="What float mode do you want to use?",
                                                    choices=[
                                                        "fp32", "bf16", "fp16"] if args["RUN_DEVICE"] == "cuda" else ["fp32", "bf16"],
                                                    )])["FLOAT_MODE"]
# print config
print("RUN_DEVICE:", args["RUN_DEVICE"])
print("FLOAT_MODE:", args["FLOAT_MODE"])
print("cudalayers:", argsnums["cudalayers"]
      if "cudalayers" in argsnums else "all")
print("")

torch.set_num_threads(12)
# opt
opt = "none"  # none // jit

if (args["RUN_DEVICE"] == "cpu" and args["FLOAT_MODE"] == "fp16"):
    raise (Warning("fp16 is only supported on cuda"))


args["MODEL_NAME"] = file
argsnums["ctx_len"] = 4068
argsnums["vocab_size"] = vocab_size
argsnums["head_qk"] = 0
argsnums["pre_ffn"] = 0
argsnums["grad_cp"] = 0
argsnums["my_pos_emb"] = 0
os.environ["RWKV_RUN_DEVICE"] = args["RUN_DEVICE"]

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'

# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese

###### A good prompt for chatbot ######
context = '''
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.

User: OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?

RWKV: It’s Joe Biden; he was sworn in earlier this year.

User: What year was the French Revolution?

RWKV: It started in 1789, but it lasted 10 years until 1799.

User: Can you guess who I might want to marry?

RWKV: Only if you tell me more about yourself - what are your interests?

User: Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?

RWKV: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.
'''
# context = "hello world! I am your supreme overlord!"
NUM_TRIALS = 999
LENGTH_PER_TRIAL = 200

TEMPERATURE = 0.9
top_p = 0.9
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

print(f'\nUsing {args["RUN_DEVICE"].upper()}. Loading {file}...')

model = RWKV_RNN(args, argsnums)
state1 = model.empty_state()

if (opt == "jit"):

    model = torch.jit.script(model)


init_state = state1


print(f'\nOptimizing speed...')
model.forward([187], state1)
gc.collect()
torch.cuda.empty_cache()

# input(0)

print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'

########################################################################################################


ctx1 = tokenizer.tokenizer.encode(context)
src_ctx1 = ctx1.copy()


print(
    "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
)

time_slot = {}
time_ref = time.time_ns()


def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


init_out = []

out = []

print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))


model.loadContext(ctx1, state1)
stateRefresh = state1.clone()


for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()
    state1 = stateRefresh.clone()
    ctx1 = src_ctx1.copy()

    if TRIAL == 0:

        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    with torch.no_grad():
        for i in range(100):

            (ctx1) = model.run(ctx1, state1, temp=TEMPERATURE, top_p=top_p)

            char = tokenizer.tokenizer.decode(ctx1[-1])

            if '\ufffd' not in char:
                print(char, end="", flush=True)

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end=''
    )

print(("-" * 50) + '\n')
