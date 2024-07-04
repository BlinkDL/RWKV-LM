#!/bin/bash

# this is a sbatch script, cf: 
#  (Choe's) https://github.com/wonkyoc/data-curation/blob/main/pretrain.slurm
#  https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output

# NB: #SBATCH are read by sbatch. the "#" symbol is needed
#SBATCH --job-name="rwkv-wc"
#SBATCH --error="rwkv.err"
#SBATCH --output="rwkv.log"
#SBATCH --mem=64G
#SBATCH -A xsel

#SBATCH --partition="interactive"
#//SBATCH --partition="gpu"
#//SBATCH --partition="standard"

#SBATCH --gres=gpu:1
#//SBATCH --gres=gpu:a100:1
#//SBATCH --gres=gpu:a100:4

#//SBATCH --constraint=gpupod

##SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=bfr4xr@virginia.edu

# env setup...
# source env-xzl.sh

echo ">>>> (FL) we are on host:" ${HOSTNAME}
# prepare a base model
# bash prep.sh

# train
# bash run.sh
