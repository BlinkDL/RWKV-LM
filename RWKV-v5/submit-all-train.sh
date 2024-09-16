OUTDIR=/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/

RUNS=
# RUNS+="04b-pre-x59 "      # pretty much done
# RUNS+="04b-pre-x59-16x "
RUNS+="1b5-pre-x59 "
RUNS+="1b5-tunefull-x58 "
RUNS+="3b-pre-x52 "
RUNS+="3b-pre-x59 "
RUNS+="3b-pre-x59-16x "
RUNS+="3b-tunefull-x58 "

for R in $RUNS; do 
    cd $OUTDIR/$R
    ./submit-train.sh
done
    