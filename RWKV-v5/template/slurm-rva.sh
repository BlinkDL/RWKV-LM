#!/bin/bash

# this is a sbatch script, based on 
#  (Choe's) https://github.com/wonkyoc/data-curation/blob/main/pretrain.slurm

#  cf: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/

# list is RV gpus
# https://www.rc.virginia.edu/userinfo/hpc/overview/


#
#    No shell script before #SBATCH -- will cause errors (why??
#

# NB: #SBATCH are read by sbatch. the "#" symbol is needed
#SBATCH --job-name="pre-01b"     #  keep it under 8 chars
#SBATCH --error="slurm.err"
#SBATCH --output="slurm.log"
#SBATCH -A xsel

#####################
#   partition related 
#####################
#SBATCH --partition="gpu"
#\\SBATCH --partition="standard"

#####################
#   gpu related 
#####################
## 4 gpus easier to get than 8 gpus
#SBATCH --gres=gpu:4
#\\SBATCH --gres=gpu:1

#\\SBATCH --gres=gpu:a100:8
#\\SBATCH --gres=gpu:a100:4
#\\SBATCH --gres=gpu:a100:1
#\\SBATCH --gres=gpu:a40:8
#\\SBATCH --gres=gpu:a6000:8
#\\SBATCH --gres=gpu:v100:4

#\\SBATCH --constraint=a100
#SBATCH --constraint=a100_80gb|a100|a6000|a40
# gpupod, https://www.rc.virginia.edu/userinfo/hpc/basepod/ only has a100 80GB
#\\SBATCH --constraint=gpupod

#####################
#   cpu related 
#####################
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2

#####################
# time limit. default 5hrs, max 3 days
#SBATCH --time=3-00:00:00

#\\SBATCH --mail-type=BEGIN,END
#\\SBATCH --mail-type=END
#\\SBATCH --mail-user=xl6yq@virginia.edu

# echo ">>>> (FL) we are on host:" ${HOSTNAME}
nvidia-smi

#####################
#   actual code 
#####################
# env setup...
RWKVROOT=`readlink -f ../../`
source $RWKVROOT/env-rivanna.sh

### prepare a base model
# bash prep.sh
#### train
bash run.sh

####################################
# useful 
####################################
## interactive shell, limited to 30 min 
# (only 2080 available, can be seen from `sinfo` below
# ijob -A xsel -p interactive --time=0-00:30:00 --cpus-per-task=8 --gres=gpu:rtx2080:1 --mem=128G 
# ijob -A xsel -p interactive --time=0-00:30:00 --gres=gpu:1 --mem=128G 
# 

## view active jobs
# squeue -u $USER

## cancel 
# scancel -u $USER      # all my jobs
# scancel 123   # job 123

## check node availability
# sinfo -p gpu -o "%20N %10R %10e %25f %25G %t %C" -t IDLE,MIX

## job accnt (also show # of cpus allocated)
# sacct

## group balance
# allocations -a xsel
