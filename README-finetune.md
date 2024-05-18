Finetune instructions

~~~~~~~~~~~~~~~~~~~~~~~~
hardware: 
preferred gpusrv with rtx2080ti (~11G VRAM) > quadro 4000 (8G VRAM)
or SLURM gpu servers....
~~~~~~~~~~~~~~~~~~~~~~~~
all code under RWKV-v5/
~~~~~~~~~~~~~~~~~~~~~~~~

1. transforming a pretrained model to decomposed weights, 
cf: svd.py 
    a pretrained model loaded from /bigtemp/ and save to /bigtemp/. see the code 
    
2. 
after saving the decomposed weights, e.g. 
    RWKV-5-World-0.4B-v2-20231113-ctx4096-svd-F4.pth

mkdir 
    RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attTune

    the dir name must match the model arch (expected by model.py)
        e.g. in the above ex, 24 layers, feature dim=1024, SVDFAC=4 
            "x052attTune" is the model variant name in model.py

copy the above *.pth as
    RWKV-LM/RWKV-v5/out/L24-D1024-F4-x052attTune/rwkv-init.pth
(rwkv-init.pth will be the initial checkpointing for finetuning) 

3. change tune.sh as needed, esp N_LAYER, N_EMBD, SVDFAC must match the model arch

    GPU_PER_NODE=1 
        for reason TBD, >1 GPUs causes exploded gradients (Nan)

4. 
    source env-xzl.sh
    tune.sh

    then use wandb to monitor training progress....
    esp look out for Nan graidents (training failures)







