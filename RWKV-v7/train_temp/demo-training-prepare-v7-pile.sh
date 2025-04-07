#!/bin/bash
#######################################################################################################################
#
# This will generate the initial model, and save it to the output folder
#
# get these files first:
# pile_20B_tokenizer_text_document.bin (664230651068 bytes)
# pile_20B_tokenizer_text_document.idx (4212099722 bytes)
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="12"
N_EMBD="768"
#
CTX_LEN="4096" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
#######################################################################################################################
#
python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "/mnt/nvme0n1/pile/pile_20B_tokenizer_text_document" --data_type "binidx" --vocab_size 50304 --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1
