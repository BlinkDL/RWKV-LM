# sync RWKV-LM/*  with remote 

# HOST=granger1
# HOST=gpusrv14
# DESTROOT=/tmp

HOST=amd2
DESTROOT=/home/xl6yq/workspace-rwkv

# RVA.....
# HOST=hpc
# DESTROOT=/home/xl6yq/workspace-rwkv

# DRYRUN=--dry-run
DRYRUN=

set -x

rsync -avXP \
    $DRYRUN \
    --exclude='out/' \
    --exclude='ChatRWKV/' \
    --exclude='staged/' \
    --exclude='*.png' \
    --exclude='RWKV-v5/wandb' \
    --exclude='orig-RWKV-v5/' \
    `pwd`   \
    $HOST:$DESTROOT

#    --exclude='RWKV-v5/data' \
#    --exclude='.git/' \