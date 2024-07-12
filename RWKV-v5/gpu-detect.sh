# https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
# https://enterprise-support.nvidia.com/s/article/Useful-nvidia-smi-Queries-2

# goal: set two env vars: 
# VRAM_MB, NGPUS

if [ $HOSTNAME = "xsel01" ]; then
    NGPUS=1; GPUID=0
elif [ $HOSTNAME = "xsel02" ]; then 
    NGPUS=1; GPUID=0
else
    NGPUS=`nvidia-smi  --list-gpus |wc -l`; GPUID=0
fi    

VRAM_MB=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$GPUID`

GPU0_NAME=`nvidia-smi --query-gpu=name --format=csv,noheader,nounits --id=$GPUID`
