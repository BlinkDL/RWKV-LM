'''
test rwkv inference engine
cf: https://pypi.org/project/rwkv/

'''
import sys, os

# run chat app on the inference engine (rwkv), check for sanity 

# xzl: use our own version of lm_eval, rwkv
sys.path.append('/home/xl6yq/workspace-rwkv/RWKV-LM')

os.environ["RWKV_JIT_ON"] = '1'

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


# rva
# model_path='/scratch/xl6yq/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'

# amd
# model_path='/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096' # official
# model_path='/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'       # v5.0 (?
# model_path='/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-ctx2K-pile/rwkv-15'    #v5.2
#v5.9
# model_path='/data-xsel01/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-x059/rwkv-init' 
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-25'
model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-35'

print(f'Loading model - {model_path}')
model = RWKV(model=model_path, strategy='cuda fp16', verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

ctx = "\nElon Musk has"
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
print('\n')

print('-----  xzl ------- \n')
# xzl: what are thsse for??? demo cut a long prompt into pieces and feed??
out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')
