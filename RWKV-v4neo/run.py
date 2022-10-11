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

# note; you can set MODEL_NAME to your fine-tuned model
size = "large"  # tini/mini/medium/medium-ext/large/xl/xxl

if (size == "tiny"):
    MODEL_NAME = "100"
    n_layer = 12
    n_embd = 768
    ctx_len = 1024

elif (size == "mini"):
    MODEL_NAME = '/fsx/BlinkDL/rwkv-release/RWKV-4-Pile-430M-20220808-8066'
    n_layer = 24
    n_embd = 1024
    ctx_len = 1024
elif (size == "medium"):
    MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
    n_layer = 24
    n_embd = 2048
    ctx_len = 1024
elif (size == "medium-ext"):
    MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220929-ctx4096'
    n_layer = 24
    n_embd = 2048
    ctx_len = 4096
elif (size == "large"):
    MODEL_NAME = 'RWKV-4-Pile-3B-20221005-7348'
    n_layer = 32
    n_embd = 2560
    ctx_len = 1024
elif (size == "xl"):
    MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20221004-3047'
    n_layer = 32
    n_embd = 4096
    ctx_len = 1024


# 'cpu' (already very fast) // 'cuda' // proc (faster then cpu, uses a fraction of the vram of cuda)
args["RUN_DEVICE"] = "proc"
# how many layers to offload to cuda, smaller number is slower, but uses less vram. // 0 -> n_layer // use to speed up proc as well
argsnums["cudalayers"] = 0
# fp32 // bf16 (saves VRAM, slightly less accurate) // fp16 (saves VRAM, slightly less accurate, can only be used with cuda, sometimes faster)
args["FLOAT_MODE"] = "fp16"

# none // ray(slower but may have better answers)
os.environ["rwkv_sampler"] = "ray"

# set max threads to 12

torch.set_num_threads(12)

# opt
opt = "jit"  # none // jit

if (args["RUN_DEVICE"] == "cpu" and args["FLOAT_MODE"] == "fp16"):
    print(Warning("fp16 is only supported on cuda, workarounds may be slow"))


args["MODEL_NAME"] = MODEL_NAME
argsnums["n_layer"] = n_layer
argsnums["n_embd"] = n_embd
argsnums["ctx_len"] = ctx_len
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

User: '''
# context = "hello world! I am your supreme overlord!"
NUM_TRIALS = 999
LENGTH_PER_TRIAL = 200

TEMPERATURE = 1.1
top_p = 0.9
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

print(f'\nUsing {args["RUN_DEVICE"].upper()}. Loading {MODEL_NAME}...')

model = RWKV_RNN(args, argsnums)

if (opt == "jit"):

    model = torch.jit.script(model)
    model = torch.jit.optimize_for_inference(model)
    model = model.eval()


state = torch.zeros(
    argsnums["n_layer"] * 5, argsnums["n_embd"], device="cpu" if args["RUN_DEVICE"] == "cpu" else "cuda", dtype=torch.float32 if args["FLOAT_MODE"] == "fp32" else torch.bfloat16 if args["FLOAT_MODE"] == "bf16" else torch.float16)
for i in range(argsnums["n_layer"]):
    state[5*i+4] -= 1e30
init_state = state


print(f'\nOptimizing speed...')
model.forward([187], state)
gc.collect()
torch.cuda.empty_cache()

# input(0)

print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'

########################################################################################################

if tokenizer.charMode:
    context = tokenizer.refine_context(context)
    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
else:
    ctx = tokenizer.tokenizer.encode(context)
src_len = len(ctx)
src_ctx = ctx.copy()

print("\nYour prompt has " + str(src_len) + " tokens.")
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
print(("-" * 50) + '\n' + context, end="")

print("torch.cuda.memory_allocated: %fGB" %
      (torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB" %
      (torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB" %
      (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print("--")
    time_ref = time.time_ns()
    ctx = src_ctx.copy()

    if TRIAL == 0:

        for i in tqdm(range(src_len)):
            x = ctx[: i + 1]
            if i == src_len - 1:
                init_out, init_state = model.forward(x, init_state)
            else:
                o, state = model.forward(
                    x, init_state, preprocess_only=True)
        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    out_last = src_len
    for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
        x = ctx
        x = x[-ctx_len:]

        if i == src_len:
            out = init_out.clone()
            state = init_state.clone()
        else:
            out, state = model.forward(x, state)
        if DEBUG_DEBUG:
            print("model", np.array(x), "==>", np.array(out), np.max(
                out.cpu().numpy()), np.min(out.cpu().numpy()))
        if TOKEN_MODE == "pile":
            out[0] = -99  # disable <|endoftext|>

        ttt = tokenizer.sample_logits(
            out,
            x,
            ctx_len,
            temperature=TEMPERATURE,
            top_p_usual=top_p,
            top_p_newline=top_p_newline,
        )
        if os.environ["rwkv_sampler"] == "ray":
            out1, state1 = model.forward(x+[ttt[0]], state.clone())
            ttt1 = tokenizer.sample_logits(
                out1,
                x+[ttt[0]],
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p_newline,
            )
            out2, state2 = model.forward(x+[ttt[1]], state.clone())
            ttt2 = tokenizer.sample_logits(
                out2,
                x+[ttt[1]],
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p_newline,
            )
            out3, state3 = model.forward(x+[ttt[2]], state.clone())
            ttt3 = tokenizer.sample_logits(
                out3,
                x+[ttt[2]],
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p_newline,
            )

            out00 = out[ttt[0]]
            out01 = out[ttt[1]]
            out02 = out[ttt[2]]

            # rays 1-9
            r1 = out00 + out1[ttt1[0]]
            r2 = out00 + out1[ttt1[1]]
            r3 = out00 + out1[ttt1[2]]
            r4 = out01 + out2[ttt2[0]]
            r5 = out01 + out2[ttt2[1]]
            r6 = out01 + out2[ttt2[2]]
            r7 = out02 + out3[ttt3[0]]
            r8 = out02 + out3[ttt3[1]]
            r9 = out02 + out3[ttt3[2]]

            mx1 = max(r1, r2, r3, r4, r5, r6, r7, r8, r9)

            if mx1 == r1:
                ctx += [ttt[0], ttt1[0]]
                state = state1
            elif mx1 == r2:
                ctx += [ttt[0], ttt1[1]]
                state = state1
            elif mx1 == r3:
                ctx += [ttt[0], ttt1[2]]
                state = state1
            elif mx1 == r4:
                ctx += [ttt[1], ttt2[0]]
                state = state2
            elif mx1 == r5:
                ctx += [ttt[1], ttt2[1]]
                state = state2
            elif mx1 == r6:
                ctx += [ttt[1], ttt2[2]]
                state = state2
            elif mx1 == r7:
                ctx += [ttt[2], ttt3[0]]
                state = state3
            elif mx1 == r8:
                ctx += [ttt[2], ttt3[1]]
                state = state3
            elif mx1 == r9:
                ctx += [ttt[2], ttt3[2]]
                state = state3
            l = 2
        else:
            ctx += [ttt]
            l = 1

        if tokenizer.charMode:
            char = tokenizer.itos[ttt]
            print(char, end="", flush=True)
        else:
            char = tokenizer.tokenizer.decode(ctx[-l:])
            if '\ufffd' not in char:
                print(char, end="", flush=True)
                out_last = i+1

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end=''
    )

print(("-" * 50) + '\n')
