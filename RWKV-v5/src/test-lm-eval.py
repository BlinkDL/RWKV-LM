# 
# to run: 
# python3.10 src/test-lm-eval.py

import run_lm_eval

# rva
# path='/scratch/xl6yq/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'

# amd
path='/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'
# path='/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-ctx2K-pile/rwkv-15'

res = run_lm_eval.do_eval(path)
print(res)

run_lm_eval.clean_cache()

# cached???
res = run_lm_eval.do_eval(path)
print(res)