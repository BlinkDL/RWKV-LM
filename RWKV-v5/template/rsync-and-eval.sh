RWKVROOT=`readlink -f ../../`

source $RWKVROOT/env-amd.sh
source $RWKVROOT/gpu-detect.sh

# Define variables
REMOTE_USER=xl6yq
REMOTE_HOST=hpc
REMOTE_DIR=/sfs/weka/scratch/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/1b5-tunefull-x58
LOCAL_DIR=.

# Find files modified in the last 2 hours on the remote server and sync them
#ssh ${REMOTE_USER}@${REMOTE_HOST} "find ${REMOTE_DIR} -regex ".*[0-9].pth" -type f -mmin -120" \
# | rsync -av --files-from=- ${REMOTE_USER}@${REMOTE_HOST}: ${LOCAL_DIR}

files=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "find ${REMOTE_DIR} -regex ".*[0-9].pth" -type f -mmin -120")

for file in $files; do    
  #scp ${REMOTE_USER}@${REMOTE_HOST}:"${file}" ${LOCAL_DIR}
  rsync -avxP ${REMOTE_USER}@${REMOTE_HOST}:"${file}" ${LOCAL_DIR}
  file=`basename $file`
  file=`readlink -f $file`
  pushd .
    cd $RWKVROOT
    python3.10 src/test-lm-eval.py $file
  popd 
done