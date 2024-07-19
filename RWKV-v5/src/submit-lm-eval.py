# 
# https://pypi.org/project/simple-slurm/

import os
import datetime
from simple_slurm import Slurm
# import run_lm_eval

# https://github.com/amq92/simple_slurm/blob/master/simple_slurm/arguments.txt

slurm = Slurm(
    account='xsel',
    partition='gpu',
    gres=['gpu:1'],
    constraint=['a100|a100_80gb'],
    job_name='lm-eval',
    output=f'slurm-eval.log',
    error=f'slurm-eval.err',
    cpus_per_task=2,
    mem='128G',
    time=datetime.timedelta(days=0, hours=2, minutes=0, seconds=0)
)

slurm.add_cmd('''RWKVROOT=`readlink -f .`''')
slurm.add_cmd('''source $RWKVROOT/env-rivanna.sh''')
slurm.add_cmd('''cd $RWKVROOT''')
slurm.sbatch('''python3.10 src/test-lm-eval.py''')
# print(slurm)