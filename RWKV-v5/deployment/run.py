########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

os.environ["RWKV_JIT_ON"] = '1'
RWKV_HOME = os.environ.get('RWKV_HOME')

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from lm_eval import tasks, evaluator
from lm_eval.api.model import TemplateLM

########################################################################################################

MODEL_NAME=f'models/04b-x59'
CLS_MODEL_NAME=f'models/04b-x59-cls.npy'

eval_tasks = ['lambada_openai']
logitBuf = {}
correctBuf = {}

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class EvalHarnessAdapter(TemplateLM):
    def __init__(self, pretrained, pipeline, pad):
        super().__init__()
        self.pad = pad
        self.pretrained = pretrained
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)

    # xzl: implement the request type "loglikelihood"....
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        # xzl: invoke forward() for each "request"... 
        # a request - a text sequent (like prompt?). eg openai benchmark has 
        # ~5K requests
        #
        # below UNLIKE test-rwkv.chat.py: NOT using pipeline args for logits sampling 
        #   just treat it as greedy decoder, for next token predict???
        for COUNTER in range(len(requests)):
            n = COUNTER
            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]

            raw_src = '\n' + raw_src
            src = self.pad + src

            sss = str(src)
            correct = True
            if sss in logitBuf: # xzl: cache...
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(self.pad)
                logit = 0
                
                with torch.no_grad():
                    print("*",end='',flush=True)   # show progress....
                    # xzl: below forward. send one seq to the model (i.e. shape bsz=1,L,D)
                    outputs, _ = self.pretrained.forward(src, None, full_output=True)
                    # breakpoint()
                    # xzl: for each token to be predicted... (q_len: prompt? 
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[i].detach().float()
                        dst = src[i+1]  # xzl: next token, from GT
                        v = F.softmax(oo, dim=-1)[dst]
                        #if v == 0:
                        #    v = 0.00000000000000000000000001
                        logit += math.log(v)    # xzl: accmulate logits for the GT token ... why?
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()   # xzl: pred token with higehst prob? greedy?
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit  # xzl: cache (accmulated) logit
                correctBuf[sss] = correct
                #clean_cache()
            
            res += [(logit, correct)]
            if n % 1000 == 0:
                print(f'{n//1000}K/{len(requests)//1000}K', end = ' ', flush=True)
        return res

    def loglikelihood_rolling():
        pass

    def generate_until():
        pass

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id


    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            #provide_description=False,
            #num_fewshot=num_fewshot,
            limit=1,
            bootstrap_iters=bootstrap_iters,
        )
        #results = evaluator.simple_evaluate(
        #    model=self,
        #    tasks=eval_tasks,
        #    num_fewshot=num_fewshot,
        #    limit=None,
        #    bootstrap_iters=bootstrap_iters,
        #    #no_cache=True
        #)
        return results

def do_eval(model_path, isverbose=False, benchmarks=[]):
    global eval_tasks

    if benchmarks==[]:
        benchmarks = eval_tasks
    # if isverbose: 
    print(f'Loading model - {model_path}')

    quant_bit = 1
    quant_map = [0.9] * 24
    mlp_map = [0.7] * 24

    # 8/26/24: using fp16 will make some benchmarks (eg openai) nan... so use fp32
    model = RWKV(model=model_path, strategy='cuda fp16', verbose=isverbose,
                 quant_bit=quant_bit, quant_map=quant_map, mlp_map=mlp_map)
    # model = RWKV(model=model_path, strategy='cuda fp32', verbose=isverbose)    # nneded for cls
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    # borrow from RWKV_CHAT
    print("\n========================================================================")
    print("Test chat...")
    args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                         alpha_frequency = 0.25,
                         alpha_presence = 0.25,
                         alpha_decay = 0.996, # gradually decay the penalty
                         token_ban = [0], # ban the generation of some tokens
                         token_stop = [], # stop generation whenever you see any token here
                         chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

    ctx = "\nAlice was so tired when she got back home so she went"
    print(ctx, end='')

    def my_print(s):
        print(s, end='', flush=True)

    pipeline.generate(ctx, token_count=50, args=args, callback=my_print)

    print("========================================================================")
    if model.stat_runs != 0:    # if model has collected any stat? print it
        print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        avg %loaded {model.stat_loaded_tokens/model.stat_runs/65535:.2f}")
        print(f"forward {model.stat_time_fwd:.2f}")
        print(f"att {model.stat_time_att:.2f}")
        print(f"ffn {model.stat_time_ffn:.2f}")
        print(f"\tmlp {model.stat_time_mlp:.2f}")
        print(f"\tquant {model.stat_time_quant:.2f}")
        print(f"\tffn: rx @ rw {model.stat_time_ffn_rx_rw:.2f}")
        print(f"\tffn: kx @ kw {model.stat_time_ffn_kx_kw:.2f}")
        print(f"\tffn: vx @ vw {model.stat_time_ffn_vx_vw:.2f}")
        print(f"cls {model.stat_time_cls:.2f}")
    print("========================================================================")

    #RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
    #if isverbose:
    #    print('RWKV_PAD', RWKV_PAD)

    #adapter = EvalHarnessAdapter(model, pipeline, RWKV_PAD)
    #results = adapter.run_eval(
    #    eval_tasks=benchmarks,
    #    bootstrap_iters=10000,
    #)

    #if model.stat_runs != 0: 
    #    print(f"stats: runs: {model.stat_runs} \
    #    cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
    #    tokens/run {model.state_loaded_tokens/model.stat_runs/65535:.2f}")
    1/0
    
    return results['results']

def clean_cache():
    global logitBuf, correctBuf
    logitBuf = {}
    correctBuf = {}

if __name__ == "__main__":
    results = do_eval(MODEL_NAME, isverbose=False)
    # print(results)
    print(json.dumps(results, indent=4, sort_keys=False))
    print("\a")   # audiable alert when done -- works on linux & Mac terminals.


