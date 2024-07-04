#!/bin/bash

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
# if [ "$VRAM_MB" -gt 10000 ] && [ "$VRAM_MB" -lt 15000 ]; then
#     M_BSZ="2"  
# fi
# # 24GB
# if [ "$VRAM_MB" -gt 20000 ] && [ "$VRAM_MB" -lt 30000 ]; then
#     M_BSZ="8"
# fi
# # 40GB
# if [ "$VRAM_MB" -gt 40000 ] && [ "$VRAM_MB" -lt 50000 ]; then
#     M_BSZ="16" # ??
# fi
# # 80 GB
# if [ "$VRAM_MB" -gt 50000 ]; then
#     M_BSZ="32" # ??
# fi

# .3B (unused)
# N_LAYER="16"
# N_EMBD="1024"

# .4B
N_LAYER="24"
N_EMBD="1024"

if [ "$VRAM_MB" -gt 10000 ] && [ "$VRAM_MB" -lt 15000 ]; then
    M_BSZ="2"  # verified
fi
# 24GB
if [ "$VRAM_MB" -gt 20000 ] && [ "$VRAM_MB" -lt 30000 ]; then
    M_BSZ="8"   # verified
fi
# 40GB
if [ "$VRAM_MB" -gt 40000 ] && [ "$VRAM_MB" -lt 50000 ]; then
    M_BSZ="10" # verified
fi
# 80 GB
if [ "$VRAM_MB" -gt 50000 ]; then
    M_BSZ="32" # ??
fi

# 1.5B
# N_LAYER="24"
# N_EMBD="2048"

# SVDFAC="16"
# SVDFAC="8"
SVDFAC="4"

HEAD_K=0    # 0 for disabling cluster head
# HEAD_K=200    # 0 for disabling cluster head

# minipile = 1.5G tokens
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 --ctx_len 512"
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 731531 --ctx_len 2048"
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 365759 --ctx_len 4096"

# pile, ~250G tokens
DATAINFO="--data_file /home/xl6yq/data/rwkv-data/uncopyright_pile/pile --my_exit_tokens 253684860910 --magic_prime 123869549 --ctx_len 2048"
# DATAINFO="--data_file /data/rwkv-data/uncopyright_pile/pile --my_exit_tokens 253684860910 --magic_prime 123869549 --ctx_len 2048"

# if [ "$MODEL_TYPE" = "x052" ]; then 
#     PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
# else 
#     PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-F"$SVDFAC"-"$MODEL_TYPE # set output folder
# fi

PROJ_DIR=`readlink -f .`

#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
# M_BSZ is per GPU, per node. so no need to scale with GPU# here
#   real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
# ctx2k, .1B, sz=8 per node --> 24GB VRAM, 
# M_BSZ="32" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
# M_BSZ="8" # ctx2k, .1B, 24GB VRAM
# M_BSZ="6" # 

# orig
#LR_INIT="6e-4"
#LR_FINAL="6e-5"

# for "pile"
LR_INIT="3e-4"
LR_FINAL="3e-5"

GRAD_CP=0 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=5 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes

# number of GPUs per node, must match the actual # of gpus...
GPU_PER_NODE=$NGPUS
# GPU_PER_NODE=4 
# GPU_PER_NODE=8 
# export CUDA_VISIBLE_DEVICES=1,2,3

# WANDB=rwkv-hpc
WANDB=

#
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)

#rm -f out/last
#ln -sf `readlink -f $PROJ_DIR` out/last

RWKVROOT=/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5
cd $RWKVROOT

python3 train.py --load_model "0" --wandb "$WANDB" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 $DATAINFO \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB \
 --svdfac $SVDFAC \
 --lm_eval_0    0