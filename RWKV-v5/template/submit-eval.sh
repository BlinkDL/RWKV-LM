export RUN_NAME=EV-`basename $PWD`   # used by: both slurm (squeue) & wandb. (cf train.py)

# CONS="a100_80gb|a100|a6000|a40"
CONS="a100_80gb|a100|a40"

sbatch --job-name=$RUN_NAME \
    --error="slurm-eval.err" \
    --output="slurm-eval.log"   \
    --account="xsel"   \
    --partition="gpu"   \
    --gres="gpu:1"      \
    --constraint=$CONS   \
    --mem="256G"  \
    --cpus-per-task=2   \
    --time=03:00:00   \
    run-eval1.sh

echo "wait for 1 sec..."; sleep 1
# https://stackoverflow.com/questions/42217102/expand-columns-to-see-full-jobname-in-slurm
#squeue --format="%.18i %.20P %.15j %.8u %.8T %R" --me
squeue --format="%.10i %.20j %.8T %20R %10M %f %b" --me