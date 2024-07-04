# https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
# https://enterprise-support.nvidia.com/s/article/Useful-nvidia-smi-Queries-2

NGPUS=`nvidia-smi  --list-gpus |wc -l`

# guess primary gpu, 0 or 1
GPUID=0
VRAM_MB=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$GPUID`

# gpu0 too small vram. must be using gpu1
if ((VRAM_MB < 4000)); then
GPUID=1
VRAM_MB=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$GPUID`
fi 




