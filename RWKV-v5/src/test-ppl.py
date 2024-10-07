########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

'''
run a model against lm benchmarks. 

make sure to use own version of lm_eval (good for cusotmization, code comment, etc

other dependencies: 
pip3 install pycountry datasets sacrebleu sqlitedict scikit-learn transformers

    alternatively, if run against the upstream lm_eval: 
    pip install rwkv lm_eval --upgrade
    newer lm_eval changed api (remove gpt2; use generic huggingface???)
    pip install rwkv lm_eval==0.3.0 

# to run: 
python3.10 src/run_lm_eval.py    
'''

import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

# xzl: use our own version of lm_eval, rwkv (update: do it in env-XXX.sh)
# sys.path.append('/home/xl6yq/workspace-rwkv/RWKV-LM')

os.environ["RWKV_JIT_ON"] = '1'

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


def is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        # Check for common Raspberry Pi hardware identifiers
        if "raspberry" in cpuinfo or any(model in cpuinfo for model in ["bcm2835", "bcm2836", "bcm2837", "bcm2711"]):
            return True
    except FileNotFoundError:
        # /proc/cpuinfo might not exist on non-Linux systems
        return False
    return False

# only do lm_eval if not pri
if not is_raspberry_pi():
    from lm_eval import tasks, evaluator
    from lm_eval.api.model import TemplateLM


########################################################################################################
MODEL_NAME = '/data/models/01b-pre-x59-CLS-TEST'
eval_tasks = ['lambada_openai']
########################################################################################################

# xzl: cached logits, cache outcome "True/False"
#       key: textual string
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

            sss = str(src)      # xzl: "src" entire sentence 
            correct = True
            # xzl cache... 
            if sss in logitBuf: # sss: a prompt processed before (cached
                logit = logitBuf[sss]       # xzl: logit for the "next token" of sss
                correct = correctBuf[sss]  # xzl: True/False for next token prediction of sss
            else:
                q_len = len(requests[n][1]) # xzl: q_len: prompt length (??) - out of the entire sentence
                q_len += len(self.pad)
                logit = 0
                
                with torch.no_grad():
                    print("*",end='',flush=True)   # show progress....
                    # xzl: below forward. send one seq to the model (i.e. shape bsz=1,L,D)
                    outputs, _, _ = self.pretrained.forward(src, None, full_output=True)
                    # breakpoint()
                    # xzl: for each token to be predicted... (q_len: prompt length?)
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[i].detach().float()
                        dst = src[i+1]  # xzl: next token, from GT
                        v = F.softmax(oo, dim=-1)[dst]

                        print(f"\nSeq length: {len(outputs)}")
                        print(f"Generated probs for the next token: {oo}")
                        print(f"\t---its length: {len(oo)}")
                        print(f"Ground truth token index: {dst}")
                        print(f"The next token logit: {v}")

                        """
                        problem:    the current "-inf" results in a zero value, which
                                    causes problems as follows:
                                    (a) math.log cannot calculate "0"
                                    (b) A way around produces a wrong ppl
                                
                        approach 1. make "v" small if v == 0 e.g.,
                        if v == 0:
                            v = 0.00000000000000000000000001

                                    this does not properly calculate ppl.

                        approach 2. fill generated probs with the minimum logits
                                    (no -inf)
                                    a. find the minimum prob of cluster in clusters
                                    b. find the minimum probs of token in the cluster
                                    c. fill this probs instead of "-inf"

                                    This does not work neither since it seems like
                                    ppl is affected by other probs...
                        """

                        # xzl: logit for the GT token. and accmulate over all preidcted tokens
                        #       for this generation process (a 'reqesut') 
                        logit += math.log(v)    
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()   # xzl: pred token with higehst prob...
                        if pred != dst:
                            correct = False     # xzl: if one token is wrong, the entire prediction is wrong
                    outputs = None
                    pred = None
                logitBuf[sss] = logit  # xzl: cache (sumed) logits for this prompt
                correctBuf[sss] = correct   # xzl: cache yes/no for this prompt
                #clean_cache()
            
            res += [(logit, correct)]   # xzl: return summed logit
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
        return results

def do_eval(model_path, isverbose=False, benchmarks=[]):
    global eval_tasks

    if benchmarks==[]:
        benchmarks = eval_tasks
    # if isverbose: 
    print(f'Loading model - {model_path}')

    # 8/26/24: using fp16 will make some benchmarks (eg openai) nan... so use fp32
    model = RWKV(model=model_path, strategy='cuda fp16', verbose=isverbose)
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

    pipeline.generate(ctx, token_count=200, args=args, callback=my_print)

    print("========================================================================")
    if model.stat_runs != 0:    # if model has collected any stat? print it
        print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        avg %loaded {model.stat_loaded_tokens/model.stat_runs/65535:.2f}")
    print("========================================================================")

    print("Start benchmark...")
    RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
    # RWKV_PAD = [0] # you can try using [0] as pad
    if isverbose:
        print('RWKV_PAD', RWKV_PAD)

    adapter = EvalHarnessAdapter(model, pipeline, RWKV_PAD)
    results = adapter.run_eval(
        eval_tasks=benchmarks,
        bootstrap_iters=10000,
    )

    print("========================================================================")
    if model.stat_runs != 0: 
        print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        tokens/run {model.stat_loaded_tokens/model.stat_runs/65535:.2f}")
    print("========================================================================")
    
    return results['results']

def clean_cache():
    global logitBuf, correctBuf
    logitBuf = {}
    correctBuf = {}

if __name__ == "__main__":

    results = do_eval(MODEL_NAME, isverbose=False)
    # print(results)
    print(json.dumps(results, indent=4, sort_keys=False))


