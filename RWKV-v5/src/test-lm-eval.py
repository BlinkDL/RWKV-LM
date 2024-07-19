# 
# to run: 
# python3.10 src/test-lm-eval.py

import os,sys
import run_lm_eval

# os.environ["RWKV_CUDA_ON"] = '0'   # for x58, x59, we dont have cuda custom ops

# rva
#v5.1
# path='/scratch/xl6yq/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'

# amd
# path='/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'       # v5.1 (?

#v5.2
#path='/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-ctx2K-pile/rwkv-15'

#v5.9
# path='/data-xsel01/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-x059/rwkv-init' 
# path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-25'  # acc .19
# path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-35'  # acc .21

#v5.8
# # path='/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pretrain-x58/from-hpc/rwkv-270-nodiag'
# # path='/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pretrain-x58/from-hpc/rwkv-295-nodiag'
path='/sfs/weka/scratch/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pretrain-x58/rwkv-410-nodiag'


if __name__ == "__main__":
    if len(sys.argv)>1:
        path=sys.argv[1]
    res = run_lm_eval.do_eval(path, isverbose=False)
    print(f"test-lm-eval {sys.argv[1]}")
    print(res)

'''
# test if res is cacahed, below 
run_lm_eval.clean_cache()

# cached???
res = run_lm_eval.do_eval(path)
print(res)

'''