rm -f slurm.err slurm.log
sbatch slurm-pretrain-rva.sh
echo "wait for 3 sec..."; sleep 3
squeue -u $USER