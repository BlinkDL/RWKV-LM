#module load nvtop
#module load cuda-toolkit-12.4.0   # needed. 
#alias python='python3'
#alias pip3='python3 -m pip'

#export CUDA_VISIBLE_DEVICES=1,2,3
#export CUDA_VISIBLE_DEVICES=0

source ${HOME}/workspace-rwkv/myenv/bin/activate

#expected by deepspeed installation 
#export CUDA_HOME=/sw/ubuntu-22.04/cuda/12.4.0/
#export CUDA_HOME=/usr/local/cuda-12/
# many things will install here
#export PATH=$PATH:${HOME}/.local/bin

export RWKVDATA=/data/rwkv-data

export RWKV_CUDA_ON=0
export PYTHONPATH=$PYTHONPATH:${HOME}/workspace-rwkv/RWKV-LM

