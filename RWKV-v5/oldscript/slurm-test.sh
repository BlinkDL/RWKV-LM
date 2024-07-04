#!/bin/bash

# this is a sbatch script
# cf: https://www.cs.virginia.edu/wiki/doku.php?id=compute_slurm
#  cf: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output

# NB: #SBATCH are read by sbatch. the "#" symbol is needed
#SBATCH --job-name="fxlin"
#SBATCH --error="slurm-test.err"
#SBATCH --output="slurm-test.log"


# list of servers: https://www.cs.virginia.edu/wiki/doku.php?id=compute_resources

# below, the order matters, otherwise sbatch will complain. the quotes also matter (?) 
#SBATCH --partition="gpu"
#SBATCH --nodelist="cheetah01,cheetah04,cheetah05,serval01,jaguar01,jaguar04"
# #of gpus needed
#SBATCH --gres=gpu:1

# first test this only
echo ">>>> (FL) we are on host:" ${HOSTNAME}

# source env-xzl.sh
# ./demo-training-run.sh

# Legacy
#export CNN_PATH=/bigtemp/slurm-demo/detectron2
#
#python $CNN_PATH/demo/demo.py \
#    --config-file "$CNN_PATH/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" \
#    --input "$CNN_PATH/demo/input1.jpg" \
#    --opts MODEL.WEIGHTS "$CNN_PATH/demo/model_final_280758.pkl"
#
