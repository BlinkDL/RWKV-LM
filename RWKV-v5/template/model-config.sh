#
MODEL_TYPE="x052" # x052 => rwkv-5.2 (rwkv-5 final)
# MODEL_TYPE="x052xzl" # my mods, both att and ffn (also dump wkv op), ffn only has key decomposed
# MODEL_TYPE="x052xzlNoReLu" # same as above, except no SqrRelu between decomposed LEFT/RIGHT matrices

# MODEL_TYPE="x052attDiag" # my mods, att only + Diag
# MODEL_TYPE="x052att" # my mods, att only

# MODEL_TYPE="x060" # x060 => rwkv-6.0
# MODEL_TYPE="mamba" # pip install mamba_ssm --upgrade

source gpu-detect.sh


# 0.1B
# N_LAYER="12"
# N_EMBD="768"
# pretrain: ctx2k, .1B, 24GB VRAM, sz=8
# 12GB
if [ "$VRAM_MB" -gt 10000 ] && [ "$VRAM_MB" -lt 15000 ]; then
    M_BSZ="2"  
fi
# 24GB
if [ "$VRAM_MB" -gt 20000 ] && [ "$VRAM_MB" -lt 30000 ]; then
    M_BSZ="8"
fi
# 40GB
if [ "$VRAM_MB" -gt 40000 ] && [ "$VRAM_MB" -lt 50000 ]; then
    M_BSZ="16" # ??
fi
# 80 GB
if [ "$VRAM_MB" -gt 50000 ]; then
    M_BSZ="32" # ??
fi

# .3B (unused)
# N_LAYER="16"
# N_EMBD="1024"

# .4B
N_LAYER="24"
N_EMBD="1024"

# 1.5B
# N_LAYER="24"
# N_EMBD="2048"

# SVDFAC="16"
# SVDFAC="8"
SVDFAC="4"

HEAD_K=0    # 0 for disabling cluster head
# HEAD_K=200    # 0 for disabling cluster head

# (not unsed)
# DATAINFO="--data_file /data/rwkv-data/data --my_exit_tokens 74958479689 --magic_prime 36600803 --ctx_len 2048"
# DATAINFO="--data_file /data/rwkv-data/data --my_exit_tokens 74958479689 --magic_prime 146403263 --ctx_len 512"

# minipile = 1.5G tokens
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 --ctx_len 512"
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 731531 --ctx_len 2048"
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 365759 --ctx_len 4096"

# pile, ~250G tokens
DATAINFO="--data_file /home/xl6yq/data/rwkv-data/uncopyright_pile/pile --my_exit_tokens 253684860910 --magic_prime 123869549 --ctx_len 2048"
# DATAINFO="--data_file /data/rwkv-data/uncopyright_pile/pile --my_exit_tokens 253684860910 --magic_prime 123869549 --ctx_len 2048"

if [ "$MODEL_TYPE" = "x052" ]; then 
    PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
else 
    PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-F"$SVDFAC"-"$MODEL_TYPE # set output folder
fi