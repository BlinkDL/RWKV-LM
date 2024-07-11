rm -f slurm.err slurm.log

export RUN_NAME=04b-pre-x52   # used by: both slurm (squeue) & wandb. (cf train.py)

sbatch --job-name=$RUN_NAME slurm-rva.sh

sbatch slurm-rva.sh
echo "wait for 1 sec..."; sleep 1
# https://stackoverflow.com/questions/42217102/expand-columns-to-see-full-jobname-in-slurm
#squeue --format="%.18i %.20P %.15j %.8u %.8T %R" --me
squeue --format="%.10i %.20j %.8T %20R %10M %f %b" --me

