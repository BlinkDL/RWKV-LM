import run_lm_eval

res = run_lm_eval.do_eval('/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-ctx2K-pile/rwkv-15')
print(res)

run_lm_eval.clean_cache()

# cached???
res = run_lm_eval.do_eval('/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-x052-ctx2K-pile/rwkv-73')
print(res)