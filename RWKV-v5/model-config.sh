#
# MODEL_TYPE="x052" # x052 => rwkv-5.2 (rwkv-5 final)
# MODEL_TYPE="x052xzl" # my mods, both att and ffn (also dump wkv op), ffn only has key decomposed
MODEL_TYPE="x052xzlNoReLu" # same as above, except no SqrRelu between decomposed LEFT/RIGHT matrices

# MODEL_TYPE="x052attDiag" # my mods, att only + Diag
# MODEL_TYPE="x052att" # my mods, att only

# MODEL_TYPE="x060" # x060 => rwkv-6.0
# MODEL_TYPE="mamba" # pip install mamba_ssm --upgrade

# 0.1B
N_LAYER="12"
N_EMBD="768"

# .3B
# N_LAYER="16"
# N_EMBD="1024"

# 1.5B
# N_LAYER="24"
# N_EMBD="2048"

# SVDFAC="16"
# SVDFAC="8"
SVDFAC="4"

#
CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-F"$SVDFAC"-"$MODEL_TYPE # set output folder
#