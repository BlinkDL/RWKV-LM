Pretrain instructions
---------------------------

# cr the result dir, e.g. name it as 01b-pre-x52. note the dir name, which 
will automaticaly become SLURM job name -- easier to find 

mkdir out/01b-pre-x52
cd out/01b-pre-x52

# copy all needed scripts to the results dir 
cp ../template/*.sh . 

# Files to change: 
# model-config.sh for model arch 
# submit-train.sh for slurm configs, e.g. # and types of gpus 
# run-train.sh for training hyperparams e.g. learning rate, etc

# prep 
grab a gpu node for interactive job 







---------------------------

Finetune instructions

---------------------------

hardware: 
preferred gpusrv with rtx2080ti (~11G VRAM) > quadro 4000 (8G VRAM)
or SLURM gpu servers....

---------------------------

all code under RWKV-v5/

---------------------------

1. transforming a pretrained model to decomposed weights, 
cf: svd.py (UNDERSTAND THIS CODE)
    a pretrained model loaded from /bigtemp/ and save to /bigtemp/. see the code 
    
2. after saving the decomposed weights, e.g. 
    RWKV-5-World-0.4B-v2-20231113-ctx4096-svd-F4.pth

mkdir RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attTune

the dir name must match the model arch (expected by model.py)

e.g. in the above ex, 24 layers, feature dim=1024, SVDFAC=4 
    "x052attTune" is the model variant name in model.py

copy the above *.pth as

    RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attTune/rwkv-init.pth

(rwkv-init.pth will be the initial checkpointing for finetuning) 

3. change tune.sh as needed, esp N_LAYER, N_EMBD, SVDFAC must match the model arch

GPU_PER_NODE=1 
    for reason TBD, >1 GPUs causes exploded gradients (Nan)

4. run
   
source env-xzl.sh

tune.sh

then use wandb to monitor training progress....
esp look out for Nan graidents (training failures)







