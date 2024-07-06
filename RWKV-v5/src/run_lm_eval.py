########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# xzl: newer lm_eval changed api (remove gpt2; use generic huggingface???)
# pip install rwkv lm_eval==0.3.0 

# we can also use own version of lm_eval (good for cusotmization, code comment, etc

# pip install pycountry datasets sacrebleu sqlitedict scikit-learn transformers

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
from lm_eval.models.gpt2 import GPT2LM

########################################################################################################

# MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

# acc .38
# MODEL_NAME = "/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096"

# acc .55 (lmo
# MODEL_NAME = "/data/models/RWKV-5-World-0.4B-v2-20231113-ctx4096"

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
# MODEL_NAME = '/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052/rwkv-73'

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

##### 1.5B 
# pretrained, very bad
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D2048-F4-x052xzlNoReLu/rwkv-11-recover"
#finetuned, acc=.49 (lmo
# MODEL_NAME = "/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L24-D2048-F4-x052xzlTune/rwkv-16-recover"

########################################################################################################

eval_tasks = []
eval_tasks += ['lambada_openai']  # OK
# eval_tasks += ['hellaswag','winogrande']   #OK, but a long test
# eval_tasks += ['lambada_openai','piqa','storycloze_2016','hellaswag','winogrande']
# eval_tasks += ['arc_challenge','arc_easy','headqa','openbookqa','sciq']
# eval_tasks += ['record','copa']

# FileNotFoundError: [Errno 2] No such file or directory: '/home/xl6yq/.cache/huggingface/datasets/downloads/extracted/639e03d5bf552352817e3dc02799cea2906076df9fe1cac49fe359a64dca8e9c/unfiltered-web-train.jsonl'
# the downloaded zip misses some file???
# need jsonl file... which we dont have 
# eval_tasks += ['triviaqa']

# this is greedy_until request...
#      max_length() not implemented 
# eval_tasks += ['coqa']

#  for 'glue', must specify separate names
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

class EvalHarnessAdapter(GPT2LM):
    def __init__(self, model, pipeline, pad):
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)
        self.model = model
        self.pad = pad

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

        # xzl: invoke forward() for each "request"... (no batching??
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
                    # print("*",end='')
                    outputs, _ = self.model.forward(src, None, full_output=True)
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[i].detach().float()
                        dst = src[i+1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            
            res += [(logit, correct)]
            if n % 1000 == 0:
                print(f'{n//1000}/{len(requests)//1000}', end = ' ', flush=True)
        return res

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=bootstrap_iters,
        )
        # results = evaluator.simple_evaluate(
        #     model=self,
        #     tasks=eval_tasks,
        #     num_fewshot=num_fewshot,
        #     limit=None,
        #     bootstrap_iters=bootstrap_iters,
        #     no_cache=True
        # )
        return results

def do_eval(model_path, isverbose=False):
    global eval_tasks

    print(f'Loading model - {model_path}')
    rwkv_model = RWKV(model=model_path, strategy='cuda fp16', verbose=isverbose)
    pipeline = PIPELINE(rwkv_model, "rwkv_vocab_v20230424")

    RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
    # RWKV_PAD = [0] # you can try using [0] as pad
    print('RWKV_PAD', RWKV_PAD)

    adapter = EvalHarnessAdapter(rwkv_model, pipeline, RWKV_PAD)
    # breakpoint()
    results = adapter.run_eval(
        eval_tasks=eval_tasks,
        bootstrap_iters=10000,
    )
    # results ex: 
    # {'lambada_openai': {'ppl': 185.72807052650887, 'ppl_stderr': 8.49129898264202, 'acc': 0.19289734135455075, 'acc_stderr': 0.005497175253106871}}
    # print(results['results'])
    return results['results']    

def clean_cache():
    global logitBuf, correctBuf
    logitBuf = {}
    correctBuf = {}

if __name__ == "__main__":
    results = do_eval(MODEL_NAME)
    print(results['results'])
