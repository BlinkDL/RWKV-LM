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

# official
# model_path='/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096' # official, NB it's v1
# model_path='/data/models/RWKV-5-World-0.4B-v2-20231113-ctx4096'

# .1B 16x, deeply compressed 
# model_path='/data/models/01b-pre-x59-16x-901'

#v5.9
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-init'   #unmodified model,  pretrained by us 
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-25'
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-35'

# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run1/rwkv-7'  # old
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-init'
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run2/rwkv-24'  #Only head.l1 tuned

# #Only head.l1 tuned. KL loss (good
model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43'

#model_path='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43'
#model_path='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-pre-x59-8x-cls/from-hpc/rwkv-1366'
#model_path='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-pre-x59-8x-cls/from-hpc/0.1b-official'
# only head.l1fc1, head.l1fc2 (MLP) trained. KL loss
#   very bad
# model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run5-KL-loss-MLP-KaimingInit/rwkv-230'
#   very bad
# model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run4-KL-loss-MLP/rwkv-40'


print(f'Loading model - {model_path}')

# xzl: for strategy, cf: https://pypi.org/project/rwkv/
#
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#

model = RWKV(model=model_path, 
             strategy='cuda fp16', 
            # strategy='cuda fp16i8',       # xzl: TBD
             verbose=True)
#              head_K=200, load_token_cls='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/from-hpc/rwkv-823-cls.npy')

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

if model.stat_runs != 0:
    print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        tokens/run {model.state_loaded_tokens/model.stat_runs/65535:.2f}")
      
'''
# xzl: what are thsse for??? demo cut a long prompt into pieces and feed??
out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')
'''