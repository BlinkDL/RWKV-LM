make sure to use own version of lm_eval (good for cusotmization, code comment, etc

other dependencies: 
pip3 install pycountry datasets sacrebleu sqlitedict scikit-learn transformers

=== Key inference code
RWKV-LM/rwkv/model.py 
    this file has comments. 
    for compressed classfication head, cf: line ~1700
    
=== Eval a model via chat (sanity check
python3.10 src/test-rwkv-chat.py

=== Eval a model against lm benchmarks
python3.10 src/run_lm_eval.py
For details, see comments at the top of run_lm_eval.py
To change models/benchmarks, cf the code 

=== Submit eval jobs on Rivanna 

cd RWKV-LM/RWKV-v5/out
mkdir 01b-pretrain-x52

cd 01b-pretrain-x52
cp ../../template/*.sh . 

./submit-eval.sh
Results will be written to eval_log.txt, eval_log.png
