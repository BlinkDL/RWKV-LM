1. install lm-evaluation-harness


./install-lm-eval.sh

This will install everything related to lm-eval including dependent libraries.

2. Check a commit
My commit is "330d91d" and the changed files are `rwkv/model.py` and `RWKV-v5/src/run_lm_eval.py`.
In `rwkv/model.py`, I added `def _retrieve_value(x ,w)` to handle multiple dimensions of `x`.
In `run_lm_eval.py`, I added more codes for compatibility with a new version of lm-eval

details:
    - A few abstract functions has been added to class EvalHarnessAdapter 
    - A few minor arugments has been changed in evaluate()

3. Run a script as-is

python run_lm_eval.py or ./submit-eval.sh (or ./submit-all-evals.sh)




