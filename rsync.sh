# HOST=granger1
# HOST=gpusrv14
# DESTROOT=/tmp

HOST=ubuntu@150.136.115.69
# DESTROOT=/home/ubuntu/workspace-rwkv-tx
DESTROOT=/home/ubuntu/workspace-rwkv

# DRYRUN=--dry-run
DRYRUN=

set -x

rsync -avXP \
    $DRYRUN \
    --exclude='out/' \
    --exclude='ChatRWKV/' \
    --exclude='staged/' \
    --exclude='.git/' \
    --exclude='*.png' \
    --exclude='RWKV-v5/wandb' \
    --exclude='RWKV-v5/data' \
    --exclude='orig-RWKV-v5/' \
    `pwd`   \
    $HOST:$DESTROOT