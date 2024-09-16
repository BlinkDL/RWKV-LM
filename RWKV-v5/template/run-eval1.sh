#!/bin/bash

# run lm-eval on all **unevaluated** chkpt (based on eval_log.txt)

# this script only contains executable commands; No slurm configs etc

#####################
#   actual code 
#####################
# env setup...
RWKVROOT=`readlink -f ../../`
OUTDIR=$PWD

if [[ $HOSTNAME == *"xsel0"* ]]; then 
    source $RWKVROOT/env-amd.sh
elif [[ $HOSTNAME == *"udc-"* ]]; then 
    source $RWKVROOT/env-rivanna.sh
fi 

nvidia-smi

#source $RWKVROOT/gpu-detect.sh
#source model-config.sh

# recent_file=$(find "." -regex ".*[0-9].pth" -type f -printf '%T+ %p\n' | sort -r | head -n 1 | awk '{print $2}')

# sort in ascending time. NB: dont recurse into sub dirs
chkpts=$(find "." -maxdepth 1 -regex ".*[0-9].pth" -type f -printf '%T+ %p\n' | sort | awk '{print $2}')

for chkpt in $chkpts; do
    bname=`basename $chkpt`
    absname=`readlink -f $chkpt`

    pushd .
    cd $RWKVROOT

    if grep -q "$bname" "$OUTDIR/eval_log.txt"; then 
        echo "(Skip $bname...)"
    else
        echo "Eval $bname..."
        # stdout also append to eval_log 
        python3.10 src/test-lm-eval.py $absname | tee -a $OUTDIR/eval_log.txt
    fi

    popd 
done

cd $RWKVROOT
python3.10 src/plot-eval.py $OUTDIR/eval_log.txt $OUTDIR/eval_log.png

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
