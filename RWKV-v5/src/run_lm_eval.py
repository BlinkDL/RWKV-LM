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


'''
# /u/xl6yq/workspace-rwkv/venv-eval-lm/lib/python3.10/site-packages/lm_eval/base.py 
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            trust_remote_code=True   # xzl
        )
'''

import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

# xzl: use our own version of lm_eval, rwkv
sys.path.append('/home/xl6yq/workspace-rwkv/RWKV-LM')

os.environ["RWKV_JIT_ON"] = '1'

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

from lm_eval import tasks, evaluator
from lm_eval.api.model import TemplateLM

########################################################################################################

# MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

# 01B --- official 
# acc .38 (openai) .29 (hellaswag) .529(winogrande  .612(piqa   ???(storycloze_2016)  .68 (copa) .67 (record, f1)
# MODEL_NAME = "/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096"

# 01B --- 01b-pre-x52
# acc:  .33 (openai) .29(hellaswag) .51(winogrande) .624 (piqa) ??(storycloze_2016) .64 (copa) .639 (record, f1)
# MODEL_NAME = '/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/from-hpc/rwkv-823'

# 01B, 01b-pre-x52 + CLS 
# baseline, .33 openai
# MODEL_NAME='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-init'  # unmodified model,  pretrained by us
# MODEL_NAME='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run/rwkv-7'  #old bad

#Only head.l1 tuned, CE loss
# MODEL_NAME='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run2/rwkv-24'  
# acc: .26 (openai)  minK=5, maxK=100, minProb=.75
# acc: .305 (openai) minK=10, maxK=150, minProb=.85
# acc: .313 (openai) minK=5, maxK=100, minProb=.95
# acc: .310 (openai) minK=3, maxK=100, minProb=.95 <--- seems good

#Only head.l1 tuned, KL loss
# acc: .331 (openai). minK=3, maxK=100, minProb=.95 <--- NEED TO CAREFULLY VERIFY
# MODEL_NAME='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43'

# 01B --- 01b-pre-x59
# acc .37 (openai) 8x (default)
# MODEL_NAME = '/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pretrain-x59/from-hpc/rwkv-976'
# 16x 
MODEL_NAME = '/data/models/01b-pre-x59-16x-901'

# 04B --- official 
# MODEL_NAME = "/data/models/RWKV-5-World-0.4B-v2-20231113-ctx4096"
'''
{'lambada_openai': {'ppl': 8.87493839489865, 'ppl_stderr': 0.25349028138862983, 'acc': 0.539879681738793, 'acc_stderr': 0.006943785077347288}, 
'hellaswag': {'acc': 0.34166500697072294, 'acc_stderr': 0.004732986187325881, 'acc_norm': 0.40908185620394344, 'acc_norm_stderr': 0.004906595857916763}, 
'winogrande': {'acc': np.float64(0.5327545382794001), 'acc_stderr': 0.014022300570434134}, 
'piqa': {'acc': 0.6653971708378672, 'acc_stderr': 0.011009071725162512, 'acc_norm': 0.6724700761697497, 'acc_norm_stderr': 0.010949830482825471}, 
'copa': {'acc': 0.69, 'acc_stderr': 0.04648231987117316}, 
'record': {'f1': 0.759412380952382, 'f1_stderr': 0.004246607860007095, 'em': 0.7527, 'em_stderr': 0.004314641655254821}}
'''

# 04B --- 04b-pre-x59 
'''
{'lambada_openai': {'ppl': 13.345917664546763, 'ppl_stderr':
    0.42075893130926734, 'acc': 0.4717640209586649, 'acc_stderr':
    0.006954861250178411}, 
'hellaswag': {'acc': 0.33270264887472617, 'acc_stderr':0.0047021810422159015, 'acc_norm': 0.3917546305516829, 'acc_norm_stderr':0.004871447106554941}, 
'winogrande': {'acc': np.float64(0.5146014206787688),
    'acc_stderr': 0.014046492383275835}, 
'piqa': {'acc': 0.676822633297062,
    'acc_stderr': 0.01091197412428213, 'acc_norm': 0.6746463547334058,
    'acc_norm_stderr': 0.010931036623525191}, 
'copa': {'acc': 0.68, 'acc_stderr': 0.046882617226215034}}
'''
# MODEL_NAME = '/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/04b-pre-x59/from-hpc/rwkv-860'

# 1B5 -- official
# MODEL_NAME = "/data/models/RWKV-5-World-1B5-v2-20231025-ctx4096"
# {'lambada_openai': {'ppl': 5.1095589778823856, 'ppl_stderr':
#   0.12199277116202499, 'acc': 0.6551523384436251, 'acc_stderr':
#   0.0066221172076032205}, 
# 'hellaswag': {'acc': 0.4237203744274049, 'acc_stderr':
#   0.0049313726571298035, 'acc_norm': 0.5500896235809599, 'acc_norm_stderr':
#   0.0049646798459184295}, 
# 'winogrande': {'acc': np.float64(0.5974743488555643),
#   'acc_stderr': 0.013782866831703044}, 
# 'piqa': {'acc': 0.7121871599564744,
#   'acc_stderr': 0.01056325038305919, 'acc_norm': 0.7225244831338411,
#   'acc_norm_stderr': 0.01044681828103994}, 
# 'copa': {'acc': 0.77, 'acc_stderr':
#   0.04229525846816506}}

# 1B5 --- 1b5-tunefull-x58
# MODEL_NAME = "/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/1b5-tunefull-x58/from-hpc/rwkv-451"

# acc ~.7
# MODEL_NAME = "/data/models/RWKV-5-World-3B-v2-20231113-ctx4096"

#0.1B
#acc 0.13, pretrained 
# MODEL_NAME = '/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-origmodel-trained-by-FL/rwkv-41'
# acc 0.18, pretrained (minipile)
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052/rwkv-final'
# large pile, pretrained. acc=0.07 (even lower??
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052/rwkv-95'   
# minipile, ctx=2K, acc=0.20, pretrained
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-ctx2048-minipile/rwkv-final'
# largepiple, ctx=2k, pretrained, acc=0 ??
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052/rwkv-15'
# uncopy pile, ctx=2k, pretrained, acc=.19
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052/rwkv-78'   #.21

###### .4B
# ok but worse acc .44 (lmo
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052xzlTune/rwkv-10-recover"
# acc .387 (overfit???
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052xzlTune/rwkv-16-recover"
# finetuned, w. Diag (acc=0.03) ... verified with chat, indeed very bad.. bug in Diag code?? (no grad)
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attDiag/rwkv-15-recover"
# acc .41 (lmo, good
# MODEL_NAME = " /data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F8-x052xzlTune/rwkv-45-recover"
# fientuned on largepile, ctx512 -- acc .334
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F8-x052xzlTune/rwkv-90-recover'
# finetuned, minipile, ctx2048 -- acc .42 (longer ctx..., note it's F8, compared to F4 above
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F8-x052xzlTune/rwkv-final-recover'

# very bad acc .03
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attTune/full-tune/rwkv-0-recover"
# acc .17 (pretrained
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-x052-orig/rwkv-45"
# acc .21 (pretrained
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D1024-x052-orig/rwkv-final"

# MODEL_NAME = '/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/04b-tunefull-x58/from-hpc/rwkv-562'

##### 1.5B 
# pretrained, very bad
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D2048-F4-x052xzlNoReLu/rwkv-11-recover"
#finetuned, acc=.49 (lmo
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D2048-F4-x052xzlTune/rwkv-16-recover"

########################################################################################################
# select benchmarks below 

eval_tasks = [
        'lambada_openai',
        
        # 'lambada_standard',
        # 'piqa',
        # 'hellaswag',
        # 'winogrande',
        # 'arc_easy',
        # 'arc_challenge',
        # 'openbookqa',
        # 'sciq',

        #'leaderboard_gpqa_main',
        #'leaderboard_ifeval',
        #'leaderboard_mmlu_pro',
        #'leaderboard_musr_murder_mysteries',
        #'leaderboard_musr_object_placements',
        #'leaderboard_musr_team_allocation',
        ]
#eval_tasks += ['lambada_openai']  # OK, (10k
#eval_tasks += ['hellaswag'] # OK (40k
#eval_tasks += ['winogrande'] # ok, fast (2k
#eval_tasks += ['piqa']  # ok, fast (3k
#eval_tasks += ['copa']  # fast <1k
# eval_tasks += ['record']  # slow 113K examples -- take long (even CUDA=1

# eval_tasks += ['storycloze_2016']  # missing in our lm_eval version. TBD

# eval_tasks += ['lambada_openai','piqa','storycloze_2016','hellaswag','winogrande']
# eval_tasks += ['arc_challenge','arc_easy','headqa','openbookqa','sciq']
# eval_tasks += ['record','copa']  # 113K examples -- take long (CUDA=1

# triviaqa 
# FileNotFoundError: [Errno 2] No such file or directory: '/home/xl6yq/.cache/huggingface/datasets/downloads/extracted/639e03d5bf552352817e3dc02799cea2906076df9fe1cac49fe359a64dca8e9c/unfiltered-web-train.jsonl'
# the downloaded zip misses some file???
# need jsonl file... which we dont have 
# eval_tasks += ['triviaqa']

# this is greedy_until request...
#      max_length() not implemented 
# eval_tasks += ['coqa']

#  for 'glue' benchmarks, must specify separate names
# eval_tasks += ['sst']  # OK 

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

    # def greedy_until(self, requests): # designed for coqa
    #     res = []
    #     for i in range(len(requests)):
    #         if i % 50 == 0:
    #             print(i)
    #         otoken = []
    #         while True:
    #             src = self.tokenizer.encode(requests[i][0]) + otoken

    #             src = src[-4096:]
    #             outputs, _ = model.forward(src, None)
                
    #             otoken += [int(torch.argmax(outputs))]
    #             ss = self.tokenizer.decode(otoken)
    #             if '\n' in ss or len(ss) > 200:
    #                 if not ss.endswith('\n'):
    #                     ss = ss + '\n'
    #                 print(ss)
    #                 res += [(ss)]
    #                 break
    #     print(res)
    #     return res

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
                        if v == 0:
                            v = 0.00000000000000000000000001
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
            limit=None,
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

def do_eval(model_path, isverbose=False):
    global eval_tasks

    # if isverbose: 
    print(f'Loading model - {model_path}')

    model = RWKV(model=model_path, strategy='cuda fp16', verbose=isverbose)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
    # RWKV_PAD = [0] # you can try using [0] as pad
    if isverbose:
        print('RWKV_PAD', RWKV_PAD)

    adapter = EvalHarnessAdapter(model, pipeline, RWKV_PAD)
    results = adapter.run_eval(
        eval_tasks=eval_tasks,
        bootstrap_iters=10000,
    )
    # results ex: 
    # {'results': {'hellaswag': {'acc': 0.2921728739294961, 'acc_stderr': 0.004538319464111977, 'acc_norm': 0.31955785700059747, 'acc_norm_stderr': 0.0046535230383693855}}, 'versions': {'hellaswag': 0}}
    # print(results['results'])

    if model.stat_runs != 0: 
        print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        tokens/run {model.state_loaded_tokens/model.stat_runs/65535:.2f}")
    
    return results['results']

def clean_cache():
    global logitBuf, correctBuf
    logitBuf = {}
    correctBuf = {}

if __name__ == "__main__":
    results = do_eval(MODEL_NAME, isverbose=False)
    # print(results)
    print(json.dumps(results, indent=4, sort_keys=False))


