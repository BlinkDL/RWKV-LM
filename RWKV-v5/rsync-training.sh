# sync current training dir to remote 

# to be executed under the specific training dir, e.g. 

# cd out/01b-x052xzl-pretrained/
# ../../sync-training.sh

# RVA.....
HOST=hpc
DESTROOT=/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out


set -x 

DRYRUN=--dry-run
# DRYRUN=

# most recent chkpt (.pth)
recent_file=$(find "." -regex ".*[0-9].pth" -type f -printf '%T+ %p\n' | sort -r | head -n 1 | awk '{print $2}')

## echo $recent_file

rsync -avXP \
    $DRYRUN \
    --include $recent_file \
    --exclude '*.pth' \
    `pwd`   \
    $HOST:$DESTROOT

