# M_BSZ="16" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
# M_BSZ="6"   # 21G, finetune .4B, ctx 2K 
# M_BSZ="4"   # 21G, finetune .4B, ctx 2K 

#
MODEL_TYPE="x052" # x052 => rwkv-5.2 (rwkv-5 final)
# MODEL_TYPE="x058" # decomposed, No relu between left/right, No diag
# MODEL_TYPE="x0585" # TBD -- decomposed, relu between left/right, No diag  -- 
# MODEL_TYPE="x059" # decomposed, relu between left/right, diag

# MODEL_TYPE="x060" # x060 => rwkv-6.0
# MODEL_TYPE="mamba" # pip install mamba_ssm --upgrade

##############################################
N_LAYER="12"; N_EMBD="768"      # 0.1B
# N_LAYER="24"; N_EMBD="1024"   # 0.4B
# N_LAYER="24"; N_EMBD="2048"

SVDFAC="8"
# SVDFAC="4"

HEAD_K=0    # 0 for disabling cluster head
# HEAD_K=200    # 0 for disabling cluster head

##########################################################################
# training data 
##########################################################################
# minipile = 1.5G tokens
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 --ctx_len 512"
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 731531 --ctx_len 2048"
# DATAINFO="--data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 365759 --ctx_len 4096"

# pile, ~250G tokens
# DATAINFO="--data_file $RWKVDATA/uncopyright_pile/pile --my_exit_tokens 253684860910 --magic_prime 123869549 --ctx_len 2048"

# dedup pile, ~200G tokens
DATAINFO="--data_file $RWKVDATA/pile_dedup/pile_dedup --my_exit_tokens 198788818379 --magic_prime 97064741 --ctx_len 2048"

##########################################################################
# proj dir
##########################################################################

TAG_FILE="L-"$N_LAYER"-D"$N_EMBD"-F"$SVDFAC"-"$MODEL_TYPE
PROJ_DIR=`readlink -f .`
touch $PROJ_DIR/$TAG_FILE-pretrain
