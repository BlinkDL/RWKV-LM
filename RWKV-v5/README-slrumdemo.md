# SLURM

## Command lists
* `sinfo`

Ex output: Note "STATE" below

```
xl6yq@portal12 (main)[RWKV-v5]$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up 4-00:00:00      8   resv cortado[02-06,08-10]
main*        up 4-00:00:00      1    mix slurm1
main*        up 4-00:00:00     10  alloc affogato[01,04-05],cortado07,lynx[08-09],slurm[2-5]
main*        up 4-00:00:00     12   idle affogato[02-03,06-10],cortado01,hydro,optane01,panther01,puma01
gpu          up 4-00:00:00     11    mix cheetah[01,04-05],jaguar[01,04-05],lynx[04,10],puma02,sds02,serval01
gpu          up 4-00:00:00     18  alloc adriatic[01-06],affogato[11-15],lynx[03,05-07,11-12],sds01
gpu          up 4-00:00:00      7   idle cheetah[02-03],jaguar[02,06],lotus,lynx[01-02]
nolim        up 20-00:00:0      1  alloc heartpiece
nolim        up 20-00:00:0      6   idle doppio[01-05],epona
gnolim       up 20-00:00:0      5    mix ai[01-04],titanx05
gnolim       up 20-00:00:0      8  alloc ai[05-10],jinx[01-02]
gnolim       up 20-00:00:0      3   idle titanx[02-04]
```

* `squeue`
Check job status, e.g. `squeue |grep $USER`
Ex
```
 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
4369358      main    fxlin    xl6yq  R       0:25      1 slurm1
```
State: R-running, PD-pending. cf: https://curc.readthedocs.io/en/latest/running-jobs/squeue-status-codes.html


* `scontrol`
* `srun`
* `sbatch`

## check SLRUM job output
will write a file like "slurm-4369350.out" in current dir, with job ID.
(there's often seconds of delay)

## SLURM on conda with 
```
# create a demo env 
conda env create -f environment.yml

# install llama.cpp
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

sbatch demo.sh
```


## Direct access to a server
This is for debugging. You shall not use it as an altertive `ssh`
```
# access to a server where your script is running
srun --nodelist ai02 --partition gnolim --pty bash -i -l -
```

