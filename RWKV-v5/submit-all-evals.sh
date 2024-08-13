OUTDIR=/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/

RUNS=
RUNS+="04b-pre-x59-16x "
RUNS+="04b-pre-x59 "
RUNS+="1b5-pre-x59 "
RUNS+="3b-pre-x52 "
RUNS+="3b-pre-x59 "
RUNS+="3b-pre-x59-16x "

for R in $RUNS; do 
    cd $OUTDIR/$R
    ./submit-eval.sh
done
