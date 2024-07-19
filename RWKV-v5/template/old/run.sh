#!/bin/bash

RWKVROOT=`readlink -f ../../`

if [[ $HOSTNAME == *"xsel0"* ]]; then 
    source $RWKVROOT/env-amd.sh
elif [[ $HOSTNAME == *"udc-"* ]]; then 
    source $RWKVROOT/env-rivanna.sh
fi 

source $RWKVROOT/gpu-detect.sh
source model-config.sh

#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#

# orig, .1B, minipile
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

# set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
if [[ $GPU0_NAME == *"A100"* ]]; then 
    DS_BUCKET_MB=200
else 
    DS_BUCKET_MB=2 
fi 

#rm -f out/last
#ln -sf `readlink -f $PROJ_DIR` out/last

cd $RWKVROOT

# for whatever reason, on RVA slurm `python3` may bind to python3.11 (why??
# so force python3.10 here...
python3.10 train.py --load_model "0" --wandb "$WANDB" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 $DATAINFO \
 --num_nodes $N_NODE --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --vram_mb $VRAM_MB \
 --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB \
 --svdfac $SVDFAC \
 --lm_eval_0    0