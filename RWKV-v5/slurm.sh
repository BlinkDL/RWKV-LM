#!/bin/bash

# this is a sbatch script
#  cf: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output

# NB: #SBATCH are read by sbatch. the "#" symbol is needed
#SBATCH --job-name="fxlin"
#SBATCH --error="demo.err"
#SBATCH --output="demo.log"


# list of servers: https://www.cs.virginia.edu/wiki/doku.php?id=compute_resources
A100=cheetah01,cheetah04,cheetah05
H100=serval01
A40=jaguar01,jaguar04
# does this work??

# below, the order matters, otherwise sbatch will complain
#SBATCH --partition="gpu"
#SBATCH --nodelist="cheetah01"
# #of gpus needed
#SBATCH --gres=gpu:1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=xl6yq@virginia.edu

# env setup...
source env-xzl.sh

echo ">>>> (FL) we are on host:" ${HOSTNAME}
# run 
./demo-training-run.sh
#sinfo


# Legacy
#export CNN_PATH=/bigtemp/slurm-demo/detectron2
#
#python $CNN_PATH/demo/demo.py \
#    --config-file "$CNN_PATH/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" \
#    --input "$CNN_PATH/demo/input1.jpg" \
#    --opts MODEL.WEIGHTS "$CNN_PATH/demo/model_final_280758.pkl"
#
