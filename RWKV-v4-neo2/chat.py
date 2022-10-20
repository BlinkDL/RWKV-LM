########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import discord
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

if (opt == "jit"):

    model = torch.jit.script(model)


print(f'\nOptimizing speed...')
gc.collect()
torch.cuda.empty_cache()

# input(0)

print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'

########################################################################################################


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


# bot.py

client = discord.Client(
    intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

# init empty save state and question context
init_state = model.empty_state()
ctx1 = tokenizer.tokenizer.encode(context)


saveStates = {}
saveStates["empty"] = ([187], init_state.clone())

# Put the prompt into the init_state
model.loadContext(ctx1, init_state)
saveStates["questions"] = (model_tokens, init_state.clone())

src_ctx1 = ctx1.copy()


@client.event
async def on_message(message):
    global model_tokens, currstate
    # print(
    #     f"message received({message.guild.name}:{message.channel.name}):", message.content)

    if message.author.bot:
        return

    msg = message.content.strip()

    if msg == '+reset_drkv' or msg == '+drkv_reset':
        model_tokens = tokenizer.tokenizer.encode(context)
        currstate = init_state

        await message.reply(f"Chat reset. This is powered by RWKV-4-{file} Language Model.")
        return

    if msg[:11] == '+drkv_save ':
        saveStates[msg[11:]] = (model_tokens, currstate)
        await message.reply(f"Saved state {msg[11:]}")
        return

    if msg[:11] == '+drkv_load ':
        if msg[11:] in saveStates:
            model_tokens, currstate = saveStates[msg[11:]]
            await message.reply(f"Loaded state {msg[11:]}")
        else:
            await message.reply(f"State {msg[11:]} not found")
        return

    if msg[:11] == '+drkv_list ':
        await message.reply(f"Saved states: {', '.join(saveStates.keys())}")
        return
    if msg[:6] == '+drkv ':

        real_msg = msg[6:].strip()
        new = f"User: {real_msg}\n\nRWKV:"
        tknew = tokenizer.tokenizer.encode(new)
        print(f'### add ###\n[{new}]')

        before = len(model_tokens)
        model_tokens = model_tokens + tknew
        begin = len(model_tokens)

        model.loadContext(model_tokens, currstate, begin=before)
        for i in tqdm(range(100)):
            (model_tokens) = model.run(model_tokens, currstate)
            if (tokenizer.tokenizer.decode(model_tokens)[-2:] == '\n\n'):
                break
        send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        print(f'### send ###\n[{send_msg}]')
        await message.reply(send_msg)

client.run(os.environ["TOKEN"])
