README-sparsity-exp.txt

9/23/2024

### data collection:

code: 
rwkv/model.py
    ffn_one_v5_9

    FFN weight is saved as a single npy "weights.npy"
    each input is saved as a tensor, which is appended to "query.npy"

run: 
RWKV-v5/src/test-rwkv-chat.py
changes prmopt and do inference 
    (slow, b/c the way we write to query.npy is inefficient) 


### mlp training, validation 
code: 
RWKV-v5/src/test-sparsity-mlp.py

    change "outpath" for loading training data, & FFN weight


### PQ
code: 
RWKV-v5/src/test-sparsity-pq.py