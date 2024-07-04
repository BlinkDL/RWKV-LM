#module load nvtop
#module load cuda-toolkit-12.4.0   # needed. 
module load cuda/12.4.1
module load gcc/11.4.0
module load anaconda
#alias python='python3'
#alias pip3='python3 -m pip'

#export CUDA_VISIBLE_DEVICES=1,2,3
#export CUDA_VISIBLE_DEVICES=0,1

# source /home/xl6yq/workspace-rwkv/venv/bin/activate

conda activate rwkv

#expected by deepspeed installation 
#export CUDA_HOME=/sw/ubuntu-22.04/cuda/12.4.0/
# export CUDA_HOME=/usr/local/cuda-12/
export CUDA_HOME=/sfs/applications/202406/software/standard/core/cuda/12.4.1/
# many things will install here
#export PATH=$PATH:${HOME}/.local/bin

export RWKVDATA=/scratch/xl6yq/data/rwkv-data