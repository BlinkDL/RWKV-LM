#!/bin/bash

BASE_NAME="model/0.1-1"
N_LAYER="12"
N_EMBD="768"
M_BSZ="16" # takes 16G VRAM (reduce this to save VRAM)
LR_INIT="6e-4"
LR_FINAL="6e-5"
GRAD_CP=0 # set to 1 to save VRAM (will be slower)
EPOCH_SAVE=10

# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case)
# use https://www.dcode.fr/prime-numbers-search

python train.py --load_model "0" --wandb "RWKV-5-Test" --proj_dir $BASE_NAME \
 --ctx_len 512 --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --num_nodes 1 --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb 200
