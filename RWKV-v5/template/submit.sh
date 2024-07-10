rm -f slurm.err slurm.log
sbatch slurm-rva.sh
echo "wait for 3 sec..."; sleep 3
# https://stackoverflow.com/questions/42217102/expand-columns-to-see-full-jobname-in-slurm
#squeue --format="%.18i %.20P %.15j %.8u %.8T %R" --me
squeue --format="%.18i %.15j %.8T %20R %10M %f %b" --me

