########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('Loading...')
from src.model_run import RWKV_RNN
import numpy as np
import os, copy, types, gc, sys
import torch
from src.utils import TOKENIZER
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
args.FLOAT_MODE = "fp16" # fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230108-5170'
args.n_layer = 40
args.n_embd = 5120
args.ctx_len = 1024

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20221115-8047'
# args.n_layer = 32
# args.n_embd = 4096
# args.ctx_len = 1024

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221008-8023'
# args.n_layer = 32
# args.n_embd = 2560
# args.ctx_len = 1024

os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
MODEL_NAME = args.MODEL_NAME

user = "User"
bot = "Bot"
interface = ":"

init_prompt = f'''
The following is a conversation between a highly knowledgeable and intelligent AI assistant called {bot}, and a human user called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} always answer {user}'s questions. {bot} is very smart, polite and humorous. {bot} knows a lot, and always tells the truth. The conversation begins.

{user}{interface} who is president of usa?

{bot}{interface} It’s Joe Biden; he was sworn in earlier this year.

{user}{interface} french revolution what year

{bot}{interface} It started in 1789, but it lasted 10 years until 1799.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} wat is lhc

{bot}{interface} It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''

# Load Model

print(f'loading... {MODEL_NAME}')
model = RWKV_RNN(args)

model_tokens = []

current_state = None

########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, current_state
    for i in range(len(tokens)):
        model_tokens += [tokens[i]]
        if i == len(tokens) - 1:
            out, current_state = model.forward(model_tokens, current_state)
        else:
            current_state = model.forward(model_tokens, current_state, preprocess_only = True)
    
    # print(f'### model ###\n[{tokenizer.tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj
    if newline_adj > 0:
        out[15] += newline_adj / 2 # '.'
    return out

all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, current_state
    n = f'{name}_{srv}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

########################################################################################################

# Run inference
print(f'\nRun prompt...')

out = run_rnn(tokenizer.tokenizer.encode(init_prompt))
gc.collect()
torch.cuda.empty_cache()

save_all_stat('', 'chat_init', out)

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{tokenizer.tokenizer.decode(model_tokens)}]\n')

def reply_msg(msg):
    print('Bot:', msg + '\n')

def on_message(message):
    global model_tokens, current_state

    srv = 'dummy_server'

    msg = message.strip()
    if len(msg) > 1000:
        reply_msg('your message is too long (max 1000 tokens)')
        return

    x_temp = 1.0
    x_top_p = 0.8
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    
    if msg == '+reset_rwkv' or msg == '+rwkv_reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:10] == '+rwkv_gen ' or msg[:9] == '+rwkv_qa ' or msg == '+rwkv_more' or msg == '+rwkv_retry' or msg == '+rwkv_again':

        if msg[:10] == '+rwkv_gen ':
            new = '\n' + msg[10:].strip()
            # print(f'### prompt ###\n[{new}]')
            current_state = None
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:9] == '+rwkv_qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[9:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

            # new = f"\nThe following is an excellent Q&A session consists of detailed and factual information.\n\nQ: What is 3+5?\nA: 3+5=8.\n\nQ: {msg[9:].strip()}\nA:"
            # print(f'### prompt ###\n[{new}]')
            # current_state = None
            # out = run_rnn(tokenizer.tokenizer.encode(new))
            # save_all_stat(srv, 'gen_0', out)

        elif msg == '+rwkv_more':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg == '+rwkv_retry' or msg == '+rwkv_again':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        for i in range(100):
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            if msg[:9] == '+rwkv_qa ':
                out = run_rnn([token], newline_adj=-2)
            else:    
                out = run_rnn([token])
        send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg == '+rwkv_alt':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(tokenizer.tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        for i in range(120):
            if i <= 0:
                newline_adj = -999999999
            elif i <= 30:
                newline_adj = -2
            elif i <= 80:
                newline_adj = 0
            elif i <= 117:
                newline_adj = (i - 80) * 0.5
            else:
                newline_adj = 999999999
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            out = run_rnn([token], newline_adj=newline_adj)
            if tokenizer.tokenizer.decode(model_tokens[-10:]).endswith(f'\n\n'):
                break

        send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

print('''Commands:
+rwkv_alt --> alternate chat reply
+rwkv_reset --> reset chat

+rwkv_gen YOUR PROMPT --> free generation with your prompt
+rwkv_qa YOUR QUESTION --> free generation - ask any question and get answer (just ask the question)
+rwkv_more --> continue last free generation [does not work for chat]
+rwkv_retry --> retry last free generation

Now talk with the bot and enjoy. Remember to +rwkv_reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +rwkv_gen for free generation.
''')

while True:
    msg = input('User: ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Erorr: please say something')
