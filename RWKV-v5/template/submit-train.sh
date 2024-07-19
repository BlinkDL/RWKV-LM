export RUN_NAME=EV-1b5-tunefull-x58   # used by: both slurm (squeue) & wandb. (cf train.py)

GPUCONFIG="gpu:4"
# GPUCONFIG="gpu:8"

# CONS="a100_80gb|a100|a6000|a40"
CONS="a100_80gb|a100"

sbatch --job-name=$RUN_NAME \
    --error="slurm.err" \
    --output="slurm.log"   \
    --account="xsel"   \
    --partition="gpu"   \
    --gres=$GPUCONFIG      \
    --constraint=$CONS   \
    --mem="256G"  \
    --cpus-per-task=2   \
    --time=3-00:00:00   \
    run.sh

echo "wait for 1 sec..."; sleep 1
# https://stackoverflow.com/questions/42217102/expand-columns-to-see-full-jobname-in-slurm
#squeue --format="%.18i %.20P %.15j %.8u %.8T %R" --me
squeue --format="%.10i %.20j %.8T %20R %10M %f %b" --me